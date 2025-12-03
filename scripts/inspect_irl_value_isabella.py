#!/usr/bin/env python3
"""
Inspect weekly reward R_theta(s_t) and discounted value V_theta(t) as an MRP.

目的：
    - 在已经学好的 IRL reward 上，显式引入一个最小的 RL 视角：
        * 把 54 个周看成一条 Markov Reward Process 轨迹；
        * 用折扣因子 gamma 计算从每一周出发的长期回报 V_theta(t)
          （往后所有周的 R_theta 加权和）；
        * 按故事阶段（valentine/art/election/recovery）查看 R / V 曲线的形状。

    - 这一步不改变任何训练，只是：
        * 给现有的 R_theta 一个「value function」解释；
        * 用 RL 语言描述“这 60 天里，不同阶段的长期情绪价值”。

用法：

    cd companion-robot-irl
    python3 scripts/inspect_irl_value_isabella.py --root data/isabella_irl_60d_openai_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataset_loader import DailyDataset
from learning.irl_preference import week_day_windows_from_sequence_length
from story.isabella_story import phase_for_day


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect IRL reward and value over weeks.")
    parser.add_argument(
        "--root",
        type=str,
        default="data/isabella_irl_60d_openai_v2",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor for value computation.",
    )
    return parser.parse_args()


def compute_discounted_values(r: np.ndarray, gamma: float) -> np.ndarray:
    """
    给定周级 reward 序列 r_t，计算从每个 t 出发的折扣回报：

        V(t) = Σ_{k=t..T-1} gamma^{k-t} * r_k

    这是一个单轨迹 MRP 上的固定策略价值函数。
    """
    T = r.shape[0]
    V = np.zeros_like(r, dtype=np.float32)
    # 从后往前动态规划。
    running = 0.0
    for t in range(T - 1, -1, -1):
        running = float(r[t]) + gamma * running
        V[t] = running
    return V


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    feat_dir = root / "features"

    # 1. 加载周级 reward：优先使用 v2（full-ranking weighted IRL），否则退回 v1。
    r_path_v2 = feat_dir / "irl_reward_weekly_v2.npy"
    r_path_v1 = feat_dir / "irl_reward_weekly.npy"
    if r_path_v2.exists():
        r = np.load(r_path_v2).astype(np.float32)
        source = "irl_reward_weekly_v2.npy"
    elif r_path_v1.exists():
        r = np.load(r_path_v1).astype(np.float32)
        source = "irl_reward_weekly.npy"
    else:
        raise FileNotFoundError(
            f"Neither irl_reward_weekly_v2.npy nor irl_reward_weekly.npy found under {feat_dir}"
        )

    T = r.shape[0]

    # 2. 生成 week -> day 范围，并用 story phase 标注。
    ds = DailyDataset(root)
    num_days = len(ds.days)
    week_day_ranges = week_day_windows_from_sequence_length(num_days, window=7, step=1)
    if len(week_day_ranges) != T:
        raise ValueError(
            f"week_day_ranges length {len(week_day_ranges)} != reward length {T}"
        )

    phases = []
    for (d_start, d_end) in week_day_ranges:
        # 该周主要阶段：简单取中点 day 的 phase。
        mid_day = (d_start + d_end) // 2
        phases.append(phase_for_day(mid_day)["name"])

    # 3. 计算折扣 value function。
    V = compute_discounted_values(r, gamma=float(args.gamma))

    print(f"[INFO] Dataset root       : {root}")
    print(f"[INFO] Reward source      : {source}")
    print(f"[INFO] Num weeks (T)      : {T}")
    print(f"[INFO] Discount factor γ  : {args.gamma}")
    print()
    print("week_idx | days      | phase            | R_theta(s_t)   | V_theta(t)")
    print("---------+-----------+------------------+---------------+-----------")
    for t in range(T):
        d_start, d_end = week_day_ranges[t]
        phase = phases[t]
        print(
            f"{t:8d} | {d_start:2d}-{d_end:2d}   | {phase:16s} | "
            f"{float(r[t]):+11.4f}   | {float(V[t]):+9.4f}"
        )

    # 4. 简单做一个按阶段的平均 R / 平均 V 汇总。
    stats = {}
    for t in range(T):
        phase = phases[t]
        stats.setdefault(phase, {"R": [], "V": []})
        stats[phase]["R"].append(float(r[t]))
        stats[phase]["V"].append(float(V[t]))

    print("\n[INFO] Phase-level summary (mean R, mean V):")
    for phase, d in stats.items():
        R_mean = float(np.mean(d["R"])) if d["R"] else float("nan")
        V_mean = float(np.mean(d["V"])) if d["V"] else float("nan")
        print(f"  - {phase:16s}: mean_R={R_mean:+.4f}, mean_V={V_mean:+.4f}")


if __name__ == "__main__":
    main()

