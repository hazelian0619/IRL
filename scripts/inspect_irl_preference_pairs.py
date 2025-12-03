#!/usr/bin/env python3
"""
Inspect weekly preference pairs for Isabella IRL V2.

用途：
    - 读取 window_valence（周级 reward）；
    - 用 learning/irl_preference.py 的策略构造偏好对 (i ≻ j)；
    - 打印若干偏好对及其 day 范围，方便和故事线做 sanity check；
    - 将 pairs 保存为 JSON 以便后续偏好 IRL 训练直接复用。

用法示例：

    cd companion-robot-irl
    python3 scripts/inspect_irl_preference_pairs.py --root data/isabella_irl_60d_openai_v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataset_loader import DailyDataset
from learning.irl_preference import (
    PreferencePairConfig,
    build_preference_pairs,
    summarize_pairs,
    week_day_windows_from_sequence_length,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect weekly preference pairs for IRL V2.")
    parser.add_argument(
        "--root",
        type=str,
        default="data/isabella_irl_60d_openai_v2",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top weeks to treat as high reward.",
    )
    parser.add_argument(
        "--bottom-k",
        type=int,
        default=5,
        help="Number of bottom weeks to treat as low reward.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.1,
        help="Additional margin for constructing extra preference pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    feat_dir = root / "features"

    # 加载周级 reward 代理 r_t。
    window_valence_path = feat_dir / "window_valence.npy"
    if not window_valence_path.exists():
        raise FileNotFoundError(f"Missing window_valence.npy under {feat_dir}")
    r = np.load(window_valence_path).astype(np.float32)  # (T_week,)

    # 用 DailyDataset 来确定总天数，并生成 day 范围。
    ds = DailyDataset(root)
    num_days = len(ds.days)
    day_windows = week_day_windows_from_sequence_length(num_days, window=7, step=1)
    if len(day_windows) != r.shape[0]:
        raise ValueError(
            f"week window count {len(day_windows)} mismatch reward length {r.shape[0]}"
        )

    cfg = PreferencePairConfig(top_k=args.top_k, bottom_k=args.bottom_k, margin=args.margin)
    pairs = build_preference_pairs(r, cfg)

    # 打印 summary，便于 story 级别检查。
    summary = summarize_pairs(r, pairs, day_windows)
    print(summary)

    # 保存 pairs 为 JSON 以供后续 IRL 训练使用。
    out_path = feat_dir / "irl_preference_pairs.json"
    json.dump(
        {"config": cfg.__dict__, "pairs": pairs},
        out_path.open("w", encoding="utf-8"),
        ensure_ascii=False,
        indent=2,
    )
    print(f"\n[INFO] Saved preference pairs ({len(pairs)}) to {out_path}")


if __name__ == "__main__":
    main()

