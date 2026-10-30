#!/usr/bin/env python3
"""Parse Bench2Drive route XML into portable scenario templates."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from common import SUPPORTED_SCENARIO_TYPES, write_json, yaw_from_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", required=True, type=Path, help="Bench2Drive route XML, e.g. leaderboard/data/bench2drive220.xml.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--types", nargs="+", default=list(SUPPORTED_SCENARIO_TYPES))
    return parser.parse_args()


def parse_float_attrs(element: ET.Element) -> dict[str, float]:
    return {key: float(value) for key, value in element.attrib.items() if key not in ("name", "type", "id", "town")}


def route_waypoints(route: ET.Element) -> list[dict[str, float]]:
    node = route.find("waypoints")
    if node is None:
        return []
    return [parse_float_attrs(position) for position in node.findall("position")]


def route_length(waypoints: list[dict[str, float]]) -> float:
    total = 0.0
    for a, b in zip(waypoints, waypoints[1:]):
        total += math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    return total


def trigger_route_s(waypoints: list[dict[str, float]], trigger: dict[str, float]) -> float:
    if not waypoints:
        return 0.0
    total = 0.0
    best_s = 0.0
    best_d = float("inf")
    last = waypoints[0]
    for index, point in enumerate(waypoints):
        if index > 0:
            total += math.hypot(point["x"] - last["x"], point["y"] - last["y"])
        dist = math.hypot(point["x"] - trigger["x"], point["y"] - trigger["y"])
        if dist < best_d:
            best_d = dist
            best_s = total
        last = point
    return best_s


def route_heading_samples(waypoints: list[dict[str, float]]) -> list[float]:
    headings = []
    for a, b in zip(waypoints, waypoints[1:]):
        headings.append(yaw_from_points(a["x"], a["y"], b["x"], b["y"]))
    return headings


def infer_topology(scenario_type: str, params: dict[str, Any]) -> list[str]:
    if scenario_type in {"PedestrianCrossing", "DynamicObjectCrossing"}:
        return ["crossing_candidate", "visible_to_front_camera"]
    if scenario_type in {"HighwayCutIn", "ParkingCutIn"}:
        return ["straight_or_gentle_curve", "adjacent_lane_preferred"]
    if scenario_type in {"ParkedObstacle", "ConstructionObstacle", "Accident"}:
        return ["straight_or_gentle_curve", "lane_or_shoulder_clearance"]
    if scenario_type == "HardBreakRoute":
        return ["straight_or_gentle_curve", "lead_vehicle_space"]
    return ["route_aligned"]


def scenario_params(scenario: ET.Element) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for child in scenario:
        if child.tag == "trigger_point":
            continue
        if "value" in child.attrib:
            raw = child.attrib["value"]
            try:
                params[child.tag] = float(raw)
            except ValueError:
                params[child.tag] = raw
        else:
            params[child.tag] = dict(child.attrib)
    return params


def parse_templates(path: Path, wanted_types: set[str]) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    templates = []
    counts: Counter[str] = Counter()
    for route in root.findall("route"):
        waypoints = route_waypoints(route)
        length = route_length(waypoints)
        headings = route_heading_samples(waypoints)
        scenarios = route.find("scenarios")
        if scenarios is None:
            continue
        for scenario in scenarios.findall("scenario"):
            scenario_type = scenario.get("type", "")
            if scenario_type not in wanted_types:
                continue
            trigger_node = scenario.find("trigger_point")
            if trigger_node is None:
                continue
            trigger = parse_float_attrs(trigger_node)
            params = scenario_params(scenario)
            trig_s = trigger_route_s(waypoints, trigger)
            templates.append(
                {
                    "template_id": f"{route.get('id')}_{scenario.get('name')}",
                    "source_route_id": route.get("id"),
                    "source_town": route.get("town"),
                    "scenario_name": scenario.get("name"),
                    "scenario_type": scenario_type,
                    "params": params,
                    "trigger_point_source": trigger,
                    "trigger_route_s": trig_s,
                    "trigger_route_fraction": trig_s / max(length, 1e-6),
                    "source_route_length": length,
                    "source_heading_start": headings[0] if headings else 0.0,
                    "source_heading_end": headings[-1] if headings else 0.0,
                    "required_topology": infer_topology(scenario_type, params),
                }
            )
            counts[scenario_type] += 1
    return {
        "source_xml": str(path),
        "supported_types": sorted(wanted_types),
        "num_templates": len(templates),
        "scenario_type_counts": dict(sorted(counts.items())),
        "templates": templates,
    }


def main() -> int:
    args = parse_args()
    payload = parse_templates(args.xml, set(args.types))
    write_json(args.output, payload)
    print(f"Wrote {args.output}")
    print(f"templates={payload['num_templates']} types={len(payload['scenario_type_counts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
