#!/usr/bin/env python3
"""
EDA and validation script for 60-day IRL datasets.

Usage:
    python scripts/eda_60d_dataset.py --root data/isabella_irl_3d_clean

This will:
    - load the dataset via DailyDataset;
    - run internal validation checks;
    - print summary statistics for mood scores and basic features.
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
from features.basic_features import build_feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA/validation on a 60-day dataset.")
    parser.add_argument(
        "--root",
        type=str,
        default="data/isabella_irl_3d_clean",
        help="Dataset root (contains conversations/behaviors/emotions/scores).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_path = Path(args.root)
    ds = DailyDataset(root_path)

    print(f"[INFO] Loaded dataset from {root_path}")
    print(f"[INFO] Days: {ds.days}")

    warnings = ds.validate()
    if warnings:
        print("\n[WARN] Dataset validation produced warnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\n[INFO] Dataset validation passed without warnings.")

    # Build basic feature matrix for quick sanity checks.
    records = [ds.get_day(d) for d in ds.days]
    feats, feat_names = build_feature_matrix(records)
    print(f"\n[INFO] Feature matrix shape: {feats.shape}")
    print(f"[INFO] Feature names: {', '.join(feat_names)}")

    # Mood score distribution.
    mood_idx = feat_names.index("mood_score")
    mood_scores = feats[:, mood_idx]
    print("\n[INFO] Mood score statistics:")
    print(f"  - min : {float(mood_scores.min()):.3f}")
    print(f"  - max : {float(mood_scores.max()):.3f}")
    print(f"  - mean: {float(mood_scores.mean()):.3f}")
    print(f"  - std : {float(mood_scores.std()):.3f}")


if __name__ == "__main__":
    main()
