# 为了测试 lane的生成情况写的

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def str_to_bool(v):
    return str(v).strip().lower() in {"true", "1", "yes", "y"}


def eval_lane_y(x, A, B, C, Dn, Df, t, fit_far_flag):
    """
    车道线公式：
    fit_far_flag = False:
        y = A + B*x + C*x^2 + Dn*x^3

    fit_far_flag = True:
        x < t:
            y = A + B*x + C*x^2 + Dn*x^3
        x >= t:
            y = A + B*x + C*x^2 + Dn*(t^3 + 3*x*(x-t)*t) + Df*(x-t)^3
    """
    if (not fit_far_flag) or x < t:
        return A + B * x + C * x * x + Dn * x * x * x

    return (
        A
        + B * x
        + C * x * x
        + Dn * (t * t * t + 3.0 * x * (x - t) * t)
        + Df * (x - t) * (x - t) * (x - t)
    )


def get_lane_prefixes(fieldnames):
    """
    自动从表头里找出 leftHost/rightHost/unknown1 这类车道线前缀。
    例如 leftHost.isValid -> leftHost
    """
    prefixes = []
    for name in fieldnames:
        if name.endswith(".isValid"):
            prefixes.append(name[: -len(".isValid")])
    return prefixes


def parse_one_lane(row, prefix, sample_step=1.0):
    """
    从一行 CSV 里面解析一条车道线，例如 leftHost。
    """
    if f"{prefix}.isValid" not in row:
        return None

    if not str_to_bool(row[f"{prefix}.isValid"]):
        return None

    try:
        fit_far_flag = str_to_bool(row[f"{prefix}.fitFarFlag"])

        A = float(row[f"{prefix}.zeroOrderCoeff"])
        B = float(row[f"{prefix}.firstOrderCoeff"])
        C = float(row[f"{prefix}.secondOrderCoeff"])
        Dn = float(row[f"{prefix}.thirdOrderCoeffNear"])
        Df = float(row[f"{prefix}.thirdOrderCoeffFar"])
        t = float(row[f"{prefix}.dividingPoint"])

        range_n = float(row[f"{prefix}.rangeN"])
        range_p = float(row[f"{prefix}.rangeP"])

        lane_width = float(row[f"{prefix}.laneWidth"])
        lane_type = int(float(row[f"{prefix}.laneType"]))
    except (KeyError, ValueError):
        return None

    if range_p <= range_n:
        return None

    # 限制采样点数量，避免极端数据导致点太多
    num = max(2, int(math.ceil((range_p - range_n) / sample_step)) + 1)
    num = min(num, 512)

    xs = np.linspace(range_n, range_p, num)
    ys = np.array([
        eval_lane_y(x, A, B, C, Dn, Df, t, fit_far_flag)
        for x in xs
    ])

    return {
        "name": prefix,
        "fit_far_flag": fit_far_flag,
        "coefficients": {
            "A_zero_order": A,
            "B_first_order": B,
            "C_second_order": C,
            "Dn_third_order_near": Dn,
            "Df_third_order_far": Df,
            "t_dividing_point": t,
        },
        "range": [range_n, range_p],
        "lane_width": lane_width,
        "lane_type": lane_type,
        "points_vehicle": [
            [float(x), float(y), 0.0]
            for x, y in zip(xs, ys)
        ],
    }


def load_frame_lanes(lane_csv, frame_num, sample_step=1.0):
    lane_csv = Path(lane_csv)

    with lane_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        prefixes = get_lane_prefixes(reader.fieldnames or [])

        for row in reader:
            try:
                cur_frame = int(row["frameNum"])
            except (KeyError, ValueError):
                continue

            if cur_frame != frame_num:
                continue

            lanes = []
            for prefix in prefixes:
                lane = parse_one_lane(row, prefix, sample_step=sample_step)
                if lane is not None:
                    lanes.append(lane)

            return {
                "frameNum": frame_num,
                "timeStamp": float(row["timeStamp"]),
                "lanes": lanes,
            }

    raise ValueError(f"frameNum={frame_num} not found in {lane_csv}")


def plot_bev(frame_data, output_png):
    plt.figure(figsize=(8, 10))

    for lane in frame_data["lanes"]:
        points = np.asarray(lane["points_vehicle"], dtype=np.float64)
        xs = points[:, 0]  # 前后方向
        ys = points[:, 1]  # 左右方向

        # BEV: 横轴画 y，纵轴画 x，这样车辆前方朝上
        plt.plot(ys, xs, label=f"{lane['name']} type={lane['lane_type']}")

    # ego vehicle 位置
    plt.scatter([0.0], [0.0], marker="x", s=80, label="ego")

    plt.xlabel("lateral y / m")
    plt.ylabel("longitudinal x / m")
    plt.title(f"Lane BEV, frameNum={frame_data['frameNum']}")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane_csv", required=True)
    parser.add_argument("--frame_num", type=int, required=True)
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--sample_step", type=float, default=1.0)
    args = parser.parse_args()

    frame_data = load_frame_lanes(
        args.lane_csv,
        args.frame_num,
        sample_step=args.sample_step,
    )

    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_bev(frame_data, output_png)

    if args.output_json is not None:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as fp:
            json.dump(frame_data, fp, indent=4)

    print(f"frameNum: {frame_data['frameNum']}")
    print(f"timeStamp: {frame_data['timeStamp']}")
    print(f"valid lanes: {len(frame_data['lanes'])}")
    for lane in frame_data["lanes"]:
        print(
            f"  {lane['name']}: "
            f"range={lane['range']}, "
            f"lane_type={lane['lane_type']}, "
            f"points={len(lane['points_vehicle'])}"
        )
    print(f"saved BEV: {output_png}")


if __name__ == "__main__":
    main()