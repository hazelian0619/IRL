#!/usr/bin/env python3
"""
Export temporal (sliding-window + decay) features for a 60-day dataset.

Usage:
    cd companion-robot-irl
    python3 scripts/export_temporal_features.py --root data/isabella_irl_3d_clean

This will populate <root>/features with:
    - rolling_stats.npy
    - global_baseline.npy
    - decay_weights.npy
    - temporal_meta.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.temporal_features import export_temporal_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export temporal features (sliding-window + decay baseline)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="data/isabella_irl_3d_clean",
        help="Dataset root (contains conversations/behaviors/emotions/scores).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_temporal_features(args.root)


if __name__ == "__main__":
    main()

