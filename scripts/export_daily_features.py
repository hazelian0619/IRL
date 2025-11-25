#!/usr/bin/env python3
"""
Export daily feature matrix and labels for 60-day IRL datasets.

Usage:
    cd companion-robot-irl
    python3 scripts/export_daily_features.py \
        --root data/isabella_irl_3d_clean \
        --out data/isabella_irl_3d_clean/features

This script will:
    - load the dataset via DailyDataset;
    - build basic daily features via build_feature_matrix;
    - save:
        - X_daily.npy          (num_days, num_features)
        - X_daily.csv          (for quick inspection)
        - y_mood_scores.npy    (num_days,)
        - feature_names.json   (list of feature names, aligned with columns)
        - days.json            (list of day indices in order)
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
from features.basic_features import build_feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export daily features for a 60-day dataset.")
    parser.add_argument(
        "--root",
        type=str,
        default="data/isabella_irl_3d_clean",
        help="Dataset root (contains conversations/behaviors/emotions/scores).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for feature files. Default: <root>/features",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    ds = DailyDataset(root)
    out_dir = Path(args.out) if args.out else (root / "features")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = [ds.get_day(d) for d in ds.days]
    X, feature_names = build_feature_matrix(records)
    y = np.array([rec.mood_score for rec in records], dtype=np.float32)

    # Save numpy arrays.
    np.save(out_dir / "X_daily.npy", X)
    np.save(out_dir / "y_mood_scores.npy", y)

    # Save CSV for quick inspection.
    csv_path = out_dir / "X_daily.csv"
    with csv_path.open("w", encoding="utf-8") as fp:
        fp.write("day," + ",".join(feature_names) + "\n")
        for day, row in zip(ds.days, X):
            values = ",".join(f"{float(v):.6g}" for v in row)
            fp.write(f"{day},{values}\n")

    # Save meta information.
    (out_dir / "feature_names.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "days.json").write_text(
        json.dumps(ds.days, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[INFO] Exported features for dataset {root}")
    print(f"  - days: {ds.days[0]}..{ds.days[-1]} (total {len(ds.days)})")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - out dir: {out_dir}")


if __name__ == "__main__":
    main()

