"""Convert nuScenes map lanes into a minimal CARLA-readable OpenDRIVE file.

This converter is intentionally conservative. nuScenes does not store a full
OpenDRIVE road model, so this script predicts the missing pieces:

* each nuScenes lane or lane_connector becomes one OpenDRIVE road;
* each road has one driving lane on the right side, with id -1;
* lane width can be fixed, or estimated from the nuScenes polygon area with a
  fixed fallback;
* lane_connector records become OpenDRIVE connecting roads inside junctions;
* successor/predecessor and junction connections are inferred from nuScenes lane
  connectivity.

The result is suitable as a first-pass topology map for CARLA/OpenDRIVE tools.
It is not a survey-grade conversion of the original nuScenes semantic map.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Point = Tuple[float, float, float]


@dataclass
class LaneRoad:
    token: str
    road_id: int
    layer: str
    centerline: List[Point]
    refline: List[Point]
    length: float
    lane_width: float
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    junction_id: Optional[int] = None

    @property
    def is_connector(self) -> bool:
        return self.layer == "lane_connector"


@dataclass
class HeightSample:
    x: float
    y: float
    z: float


class HeightSampler:
    def __init__(self, samples: Sequence[HeightSample], neighbor_count: int, search_radius: float) -> None:
        if not samples:
            raise ValueError("height pose file contains no samples")
        self.samples = list(samples)
        self.neighbor_count = max(1, int(neighbor_count))
        self.search_radius = max(0.0, float(search_radius))

    def sample(self, x_value: float, y_value: float) -> float:
        distances = []
        for sample in self.samples:
            distance = math.hypot(sample.x - x_value, sample.y - y_value)
            if distance <= 1e-9:
                return sample.z
            if self.search_radius <= 0.0 or distance <= self.search_radius:
                distances.append((distance, sample.z))

        if not distances:
            nearest = min(
                self.samples,
                key=lambda sample: math.hypot(sample.x - x_value, sample.y - y_value),
            )
            return nearest.z

        distances.sort(key=lambda item: item[0])
        nearest = distances[: self.neighbor_count]
        weights = [1.0 / max(distance, 1e-6) for distance, _z in nearest]
        total_weight = sum(weights)
        return sum(weight * z_value for weight, (_distance, z_value) in zip(weights, nearest)) / total_weight


class DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def fmt(value: float) -> str:
    return f"{value:.16e}"


def dist2d(left: Point, right: Point) -> float:
    return math.hypot(right[0] - left[0], right[1] - left[1])


def polyline_length(points: Sequence[Point]) -> float:
    return sum(dist2d(a, b) for a, b in zip(points[:-1], points[1:]))


def dedupe_points(points: Sequence[Point], min_distance: float) -> List[Point]:
    deduped: List[Point] = []
    for point in points:
        if not deduped or dist2d(deduped[-1], point) >= min_distance:
            deduped.append(point)
    if len(deduped) == 1 and len(points) > 1:
        deduped.append(points[-1])
    return deduped


def read_height_pose_csv(path: Path, origin_x: float, origin_y: float, flip_y: bool, z_offset: float) -> List[HeightSample]:
    samples: List[HeightSample] = []
    with path.open("r", encoding="utf-8") as fp:
        header_line = fp.readline().strip()
        if not header_line:
            return samples
        header = header_line.replace(",", " ").split()
        index = {name: idx for idx, name in enumerate(header)}
        missing = [name for name in ("x", "y", "z") if name not in index]
        if missing:
            raise ValueError(f"{path} is missing columns {missing}; expected at least x y z")

        for line in fp:
            if not line.strip():
                continue
            parts = line.replace(",", " ").split()
            x_value = float(parts[index["x"]]) - origin_x
            y_value = float(parts[index["y"]]) - origin_y
            if flip_y:
                y_value = -y_value
            samples.append(HeightSample(x_value, y_value, float(parts[index["z"]]) + z_offset))
    return samples


def apply_height_profile(points: Sequence[Point], sampler: Optional[HeightSampler], smoothing_window: int) -> List[Point]:
    if sampler is None:
        return list(points)

    z_values = [sampler.sample(point[0], point[1]) for point in points]
    window = max(1, int(smoothing_window))
    if window > 1 and len(z_values) > 2:
        radius = window // 2
        smoothed = []
        for index in range(len(z_values)):
            start = max(0, index - radius)
            end = min(len(z_values), index + radius + 1)
            smoothed.append(sum(z_values[start:end]) / float(end - start))
        z_values = smoothed

    return [(point[0], point[1], z_value) for point, z_value in zip(points, z_values)]


def offset_refline_from_centerline(centerline: Sequence[Point], lane_width: float) -> List[Point]:
    """Place the OpenDRIVE reference line left of the nuScenes lane centerline.

    A negative OpenDRIVE lane id travels with the reference line direction and is
    located to the right of the reference line. Offsetting by half the lane
    width therefore puts the generated lane -1 center near the nuScenes
    centerline.
    """

    if len(centerline) < 2:
        return list(centerline)

    refline: List[Point] = []
    for index, point in enumerate(centerline):
        if index == 0:
            neighbor = centerline[1]
            heading = math.atan2(neighbor[1] - point[1], neighbor[0] - point[0])
        elif index == len(centerline) - 1:
            neighbor = centerline[index - 1]
            heading = math.atan2(point[1] - neighbor[1], point[0] - neighbor[0])
        else:
            previous_point = centerline[index - 1]
            next_point = centerline[index + 1]
            heading = math.atan2(next_point[1] - previous_point[1], next_point[0] - previous_point[0])

        left_x = -math.sin(heading)
        left_y = math.cos(heading)
        refline.append((point[0] + left_x * lane_width * 0.5, point[1] + left_y * lane_width * 0.5, point[2]))

    return refline


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_connected_lanes(nusc_map, token: str, direction: str) -> List[str]:
    method_name = "get_incoming_lane_ids" if direction == "incoming" else "get_outgoing_lane_ids"
    method = getattr(nusc_map, method_name, None)
    if method is None:
        return []
    try:
        return list(method(token))
    except Exception:
        return []


def discretize_lane(nusc_map, token: str, resolution: float) -> List[Point]:
    from nuscenes.map_expansion import arcline_path_utils

    arcline_path = nusc_map.get_arcline_path(token)
    poses = arcline_path_utils.discretize_lane(arcline_path, resolution_meters=resolution)
    return [(float(x), float(y), 0.0) for x, y, _yaw in poses]


def polygon_area(polygon) -> Optional[float]:
    area = getattr(polygon, "area", None)
    if area is not None:
        return float(area)

    exterior = getattr(polygon, "exterior", None)
    coords = getattr(exterior, "coords", None)
    if coords is None:
        return None

    points = [(float(x), float(y)) for x, y, *_rest in coords]
    if len(points) < 3:
        return None

    signed = 0.0
    for left, right in zip(points, points[1:] + points[:1]):
        signed += left[0] * right[1] - right[0] * left[1]
    return abs(signed) * 0.5


def estimate_lane_width(nusc_map, record: dict, length: float, default_width: float,
                        min_width: float, max_width: float) -> float:
    polygon_token = record.get("polygon_token")
    if polygon_token and length > 0.1:
        try:
            polygon = nusc_map.extract_polygon(polygon_token)
            area = polygon_area(polygon)
            if area is not None and area > 0.1:
                return clamp(area / length, min_width, max_width)
        except Exception:
            pass
    return default_width


def lane_width_for_record(nusc_map, layer: str, record: dict, length: float, args) -> float:
    if layer == "lane_connector" and args.connector_lane_width is not None:
        return args.connector_lane_width
    if args.lane_width_mode == "fixed":
        return args.default_lane_width
    return estimate_lane_width(
        nusc_map,
        record,
        length,
        args.default_lane_width,
        args.min_lane_width,
        args.max_lane_width,
    )


def transform_points(points: Sequence[Point], origin_x: float, origin_y: float, flip_y: bool) -> List[Point]:
    transformed = []
    for x_value, y_value, z_value in points:
        x_new = x_value - origin_x
        y_new = y_value - origin_y
        if flip_y:
            y_new = -y_new
        transformed.append((x_new, y_new, z_value))
    return transformed


def center_origin(entries: Sequence[Tuple[str, str, dict, List[Point], float]]) -> Tuple[float, float]:
    xs = [point[0] for *_prefix, centerline, _length in entries for point in centerline]
    ys = [point[1] for *_prefix, centerline, _length in entries for point in centerline]
    return (min(xs) + max(xs)) * 0.5, (min(ys) + max(ys)) * 0.5


def build_roads(args) -> Tuple[Dict[str, LaneRoad], Dict[str, int]]:
    from nuscenes.map_expansion.map_api import NuScenesMap

    nusc_map = NuScenesMap(dataroot=args.dataroot, map_name=args.map_name)

    raw_records: List[Tuple[str, dict]] = []
    for layer in ("lane", "lane_connector"):
        for record in getattr(nusc_map, layer):
            raw_records.append((layer, record))

    raw_records.sort(key=lambda item: item[1]["token"])

    entries: List[Tuple[str, str, dict, List[Point], float]] = []
    skipped: Dict[str, int] = {"short": 0, "path_error": 0}

    for layer, record in raw_records:
        token = record["token"]
        try:
            centerline = discretize_lane(nusc_map, token, args.resolution)
        except Exception as exc:
            skipped["path_error"] += 1
            if args.verbose:
                print(f"skip {layer} {token}: cannot discretize lane: {exc}", file=sys.stderr)
            continue

        centerline = dedupe_points(centerline, args.min_segment_length)
        length = polyline_length(centerline)
        if len(centerline) < 2 or length < args.min_road_length:
            skipped["short"] += 1
            continue

        entries.append((token, layer, record, centerline, length))

    if args.center_map and entries:
        origin_x, origin_y = center_origin(entries)
    else:
        origin_x, origin_y = args.origin_x, args.origin_y

    height_sampler: Optional[HeightSampler] = None
    if args.height_pose_csv:
        height_samples = read_height_pose_csv(
            Path(args.height_pose_csv),
            origin_x,
            origin_y,
            args.flip_y,
            args.height_z_offset,
        )
        height_sampler = HeightSampler(height_samples, args.height_neighbor_count, args.height_search_radius)

    roads: Dict[str, LaneRoad] = {}
    for token, layer, record, centerline, native_length in entries:
        centerline = transform_points(centerline, origin_x, origin_y, args.flip_y)
        centerline = apply_height_profile(centerline, height_sampler, args.height_smoothing_window)
        lane_width = lane_width_for_record(nusc_map, layer, record, native_length, args)
        refline = offset_refline_from_centerline(centerline, lane_width)
        refline = dedupe_points(refline, args.min_segment_length)
        refline = apply_height_profile(refline, height_sampler, args.height_smoothing_window)
        length = polyline_length(refline)

        roads[token] = LaneRoad(
            token=token,
            road_id=len(roads) + 1,
            layer=layer,
            centerline=centerline,
            refline=refline,
            length=length,
            lane_width=lane_width,
            predecessors=safe_connected_lanes(nusc_map, token, "incoming"),
            successors=safe_connected_lanes(nusc_map, token, "outgoing"),
        )

    valid_tokens = set(roads)
    for road in roads.values():
        road.predecessors = [token for token in road.predecessors if token in valid_tokens]
        road.successors = [token for token in road.successors if token in valid_tokens]

    return roads, skipped


def infer_junctions(roads: Dict[str, LaneRoad], merge_distance: float) -> Dict[int, List[str]]:
    connector_tokens = [token for token, road in roads.items() if road.is_connector]
    dsu = DisjointSet(connector_tokens)

    lane_to_connectors: Dict[str, List[str]] = {}
    for token in connector_tokens:
        road = roads[token]
        for lane_token in road.predecessors + road.successors:
            lane_to_connectors.setdefault(lane_token, []).append(token)

    for connected in lane_to_connectors.values():
        for token in connected[1:]:
            dsu.union(connected[0], token)

    if merge_distance > 0.0:
        endpoints = {
            token: (roads[token].centerline[0], roads[token].centerline[-1])
            for token in connector_tokens
        }
        for index, left in enumerate(connector_tokens):
            for right in connector_tokens[index + 1:]:
                left_points = endpoints[left]
                right_points = endpoints[right]
                if any(dist2d(a, b) <= merge_distance for a in left_points for b in right_points):
                    dsu.union(left, right)

    groups_by_root: Dict[str, List[str]] = {}
    for token in connector_tokens:
        groups_by_root.setdefault(dsu.find(token), []).append(token)

    junctions: Dict[int, List[str]] = {}
    for junction_id, tokens in enumerate(groups_by_root.values(), start=1):
        for token in tokens:
            roads[token].junction_id = junction_id
        junctions[junction_id] = sorted(tokens, key=lambda token: roads[token].road_id)

    return junctions


def first_existing(tokens: Sequence[str], roads: Dict[str, LaneRoad], want_connector: bool) -> Optional[LaneRoad]:
    for token in tokens:
        road = roads.get(token)
        if road is not None and road.is_connector == want_connector:
            return road
    return None


def junction_for_connected_roads(tokens: Sequence[str], roads: Dict[str, LaneRoad]) -> Optional[int]:
    for token in tokens:
        road = roads.get(token)
        if road is not None and road.is_connector and road.junction_id is not None:
            return road.junction_id
    return None


def add_road_link(parent: ET.Element, road: LaneRoad, roads: Dict[str, LaneRoad]) -> None:
    link = ET.SubElement(parent, "link")

    if road.is_connector:
        predecessor = first_existing(road.predecessors, roads, want_connector=False)
        successor = first_existing(road.successors, roads, want_connector=False)
        if predecessor is not None:
            ET.SubElement(
                link,
                "predecessor",
                elementType="road",
                elementId=str(predecessor.road_id),
                contactPoint="end",
            )
        if successor is not None:
            ET.SubElement(
                link,
                "successor",
                elementType="road",
                elementId=str(successor.road_id),
                contactPoint="start",
            )
    else:
        predecessor_junction = junction_for_connected_roads(road.predecessors, roads)
        successor_junction = junction_for_connected_roads(road.successors, roads)
        predecessor_road = first_existing(road.predecessors, roads, want_connector=False)
        successor_road = first_existing(road.successors, roads, want_connector=False)

        if predecessor_junction is not None:
            ET.SubElement(link, "predecessor", elementType="junction", elementId=str(predecessor_junction))
        elif predecessor_road is not None:
            ET.SubElement(
                link,
                "predecessor",
                elementType="road",
                elementId=str(predecessor_road.road_id),
                contactPoint="end",
            )

        if successor_junction is not None:
            ET.SubElement(link, "successor", elementType="junction", elementId=str(successor_junction))
        elif successor_road is not None:
            ET.SubElement(
                link,
                "successor",
                elementType="road",
                elementId=str(successor_road.road_id),
                contactPoint="start",
            )


def add_plan_view(parent: ET.Element, refline: Sequence[Point], min_segment_length: float) -> None:
    plan_view = ET.SubElement(parent, "planView")
    s_offset = 0.0
    for start, end in zip(refline[:-1], refline[1:]):
        segment_length = dist2d(start, end)
        if segment_length < min_segment_length:
            continue
        heading = math.atan2(end[1] - start[1], end[0] - start[0])
        geometry = ET.SubElement(
            plan_view,
            "geometry",
            s=fmt(s_offset),
            x=fmt(start[0]),
            y=fmt(start[1]),
            hdg=fmt(heading),
            length=fmt(segment_length),
        )
        ET.SubElement(geometry, "line")
        s_offset += segment_length


def add_elevation_profile(parent: ET.Element, refline: Sequence[Point], min_segment_length: float) -> None:
    elevation_profile = ET.SubElement(parent, "elevationProfile")
    has_height_change = any(abs(point[2] - refline[0][2]) > 1e-4 for point in refline[1:]) if refline else False
    if not has_height_change:
        z_value = refline[0][2] if refline else 0.0
        ET.SubElement(
            elevation_profile,
            "elevation",
            s=fmt(0.0),
            a=fmt(z_value),
            b=fmt(0.0),
            c=fmt(0.0),
            d=fmt(0.0),
        )
        return

    s_offset = 0.0
    wrote_segment = False
    for start, end in zip(refline[:-1], refline[1:]):
        segment_length = dist2d(start, end)
        if segment_length < min_segment_length:
            continue
        ET.SubElement(
            elevation_profile,
            "elevation",
            s=fmt(s_offset),
            a=fmt(start[2]),
            b=fmt((end[2] - start[2]) / segment_length),
            c=fmt(0.0),
            d=fmt(0.0),
        )
        wrote_segment = True
        s_offset += segment_length

    if not wrote_segment:
        ET.SubElement(
            elevation_profile,
            "elevation",
            s=fmt(0.0),
            a=fmt(refline[0][2] if refline else 0.0),
            b=fmt(0.0),
            c=fmt(0.0),
            d=fmt(0.0),
        )


def add_lanes(parent: ET.Element, road: LaneRoad, roads: Dict[str, LaneRoad]) -> None:
    lanes = ET.SubElement(parent, "lanes")
    ET.SubElement(lanes, "laneOffset", s=fmt(0.0), a=fmt(0.0), b=fmt(0.0), c=fmt(0.0), d=fmt(0.0))
    lane_section = ET.SubElement(lanes, "laneSection", s=fmt(0.0))

    center = ET.SubElement(lane_section, "center")
    center_lane = ET.SubElement(center, "lane", id="0", type="none", level="false")
    ET.SubElement(center_lane, "roadMark", sOffset=fmt(0.0), type="none", material="standard", color="white")

    right = ET.SubElement(lane_section, "right")
    driving_lane = ET.SubElement(right, "lane", id="-1", type="driving", level="false")
    lane_link = ET.SubElement(driving_lane, "link")
    if any(token in roads for token in road.predecessors):
        ET.SubElement(lane_link, "predecessor", id="-1")
    if any(token in roads for token in road.successors):
        ET.SubElement(lane_link, "successor", id="-1")
    ET.SubElement(
        driving_lane,
        "width",
        sOffset=fmt(0.0),
        a=fmt(road.lane_width),
        b=fmt(0.0),
        c=fmt(0.0),
        d=fmt(0.0),
    )
    ET.SubElement(
        driving_lane,
        "roadMark",
        sOffset=fmt(0.0),
        type="broken",
        material="standard",
        color="white",
        laneChange="both",
    )


def add_road(parent: ET.Element, road: LaneRoad, roads: Dict[str, LaneRoad], min_segment_length: float) -> None:
    road_element = ET.SubElement(
        parent,
        "road",
        name=f"{road.layer}_{road.road_id}",
        length=fmt(road.length),
        id=str(road.road_id),
        junction=str(road.junction_id if road.is_connector and road.junction_id is not None else -1),
    )
    add_road_link(road_element, road, roads)
    road_type = ET.SubElement(road_element, "type", s=fmt(0.0), type="town")
    ET.SubElement(road_type, "speed", max="13.89", unit="m/s")
    add_plan_view(road_element, road.refline, min_segment_length)

    add_elevation_profile(road_element, road.refline, min_segment_length)
    lateral_profile = ET.SubElement(road_element, "lateralProfile")
    ET.SubElement(lateral_profile, "superelevation", s=fmt(0.0), a=fmt(0.0), b=fmt(0.0), c=fmt(0.0), d=fmt(0.0))
    add_lanes(road_element, road, roads)


def add_junctions(parent: ET.Element, junctions: Dict[int, List[str]], roads: Dict[str, LaneRoad]) -> None:
    for junction_id, connector_tokens in sorted(junctions.items()):
        junction = ET.SubElement(parent, "junction", id=str(junction_id), name=f"junction_{junction_id}")
        connection_id = 0
        for connector_token in connector_tokens:
            connector = roads[connector_token]
            incoming = first_existing(connector.predecessors, roads, want_connector=False)
            if incoming is None:
                continue
            connection = ET.SubElement(
                junction,
                "connection",
                id=str(connection_id),
                incomingRoad=str(incoming.road_id),
                connectingRoad=str(connector.road_id),
                contactPoint="start",
            )
            ET.SubElement(connection, "laneLink", **{"from": "-1", "to": "-1"})
            connection_id += 1


def bounds_for_roads(roads: Dict[str, LaneRoad]) -> Tuple[float, float, float, float]:
    xs = [point[0] for road in roads.values() for point in road.refline]
    ys = [point[1] for road in roads.values() for point in road.refline]
    return max(ys), min(ys), max(xs), min(xs)


def build_opendrive_xml(args, roads: Dict[str, LaneRoad], junctions: Dict[int, List[str]]) -> ET.ElementTree:
    north, south, east, west = bounds_for_roads(roads)
    root = ET.Element("OpenDRIVE")
    header = ET.SubElement(
        root,
        "header",
        revMajor="1",
        revMinor="4",
        name=args.map_name,
        version="1.00",
        date=_datetime.datetime.now().isoformat(timespec="seconds"),
        north=fmt(north),
        south=fmt(south),
        east=fmt(east),
        west=fmt(west),
        vendor="CARLA nuScenes converter",
    )
    if args.geo_reference:
        geo = ET.SubElement(header, "geoReference")
        geo.text = args.geo_reference

    for road in sorted(roads.values(), key=lambda item: item.road_id):
        add_road(root, road, roads, args.min_segment_length)

    add_junctions(root, junctions, roads)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    return tree


def validate_with_carla(xodr_path: Path) -> None:
    import carla

    xodr_data = xodr_path.read_text(encoding="utf-8")
    carla_map = carla.Map(xodr_path.stem, xodr_data)
    topology = carla_map.get_topology()
    print(f"CARLA validation passed: topology edges={len(topology)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", required=True, help="nuScenes dataroot containing maps/ and samples/.")
    parser.add_argument(
        "--map-name",
        required=True,
        choices=["boston-seaport", "singapore-hollandvillage", "singapore-onenorth", "singapore-queenstown"],
        help="nuScenes map name.",
    )
    parser.add_argument("--output", required=True, help="Output .xodr path.")
    parser.add_argument("--resolution", type=float, default=1.0, help="Lane centerline sampling resolution in meters.")
    parser.add_argument("--default-lane-width", type=float, default=3.5, help="Fallback lane width in meters.")
    parser.add_argument(
        "--lane-width-mode",
        choices=["estimated", "fixed"],
        default="estimated",
        help="Use nuScenes polygon area to estimate widths, or force every lane to --default-lane-width.",
    )
    parser.add_argument(
        "--connector-lane-width",
        type=float,
        default=None,
        help="Optional fixed width for lane_connector roads. Useful for reducing junction overlap.",
    )
    parser.add_argument("--min-lane-width", type=float, default=2.5, help="Clamp estimated lane width lower bound.")
    parser.add_argument("--max-lane-width", type=float, default=5.0, help="Clamp estimated lane width upper bound.")
    parser.add_argument("--min-road-length", type=float, default=1.0, help="Skip lanes shorter than this many meters.")
    parser.add_argument(
        "--min-segment-length",
        type=float,
        default=0.25,
        help="Skip or merge line geometry segments shorter than this many meters.",
    )
    parser.add_argument(
        "--junction-merge-distance",
        type=float,
        default=8.0,
        help="Merge lane_connectors into one junction when their endpoints are within this distance.",
    )
    parser.add_argument(
        "--geo-reference",
        default="",
        help="Optional PROJ string for the OpenDRIVE header geoReference.",
    )
    parser.add_argument("--origin-x", type=float, default=0.0, help="Subtract this nuScenes x coordinate.")
    parser.add_argument("--origin-y", type=float, default=0.0, help="Subtract this nuScenes y coordinate.")
    parser.add_argument("--center-map", action="store_true", help="Center converted lane coordinates around (0, 0).")
    parser.add_argument("--flip-y", action="store_true", help="Mirror y after origin subtraction.")
    parser.add_argument(
        "--height-pose-csv",
        type=Path,
        help=(
            "Optional whitespace- or comma-separated pose file with at least x/y/z columns. "
            "When omitted, the converter writes the original flat z=0 XODR."
        ),
    )
    parser.add_argument(
        "--height-neighbor-count",
        type=int,
        default=4,
        help="Number of nearby pose samples used for inverse-distance height interpolation.",
    )
    parser.add_argument(
        "--height-search-radius",
        type=float,
        default=0.0,
        help="Only use pose samples within this XY radius in meters. 0 means use nearest samples globally.",
    )
    parser.add_argument(
        "--height-smoothing-window",
        type=int,
        default=5,
        help="Moving-average window over per-road z samples. 1 disables smoothing.",
    )
    parser.add_argument(
        "--height-z-offset",
        type=float,
        default=0.0,
        help="Constant offset added to every imported height sample.",
    )
    parser.add_argument("--validate-carla", action="store_true", help="Try to parse the generated file with carla.Map.")
    parser.add_argument("--verbose", action="store_true", help="Print skipped lane details.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)

    roads, skipped = build_roads(args)
    if not roads:
        print("no valid nuScenes lanes were converted", file=sys.stderr)
        return 2

    junctions = infer_junctions(roads, args.junction_merge_distance)
    tree = build_opendrive_xml(args, roads, junctions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)

    ET.parse(output_path)
    print(
        f"wrote {output_path} roads={len(roads)} junctions={len(junctions)} "
        f"skipped_short={skipped['short']} skipped_path_error={skipped['path_error']}"
    )

    if args.validate_carla:
        validate_with_carla(output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
