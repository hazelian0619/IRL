#!/usr/bin/env python3
"""
Export weekly preference features for IRL V2.

用法示例：

    cd companion-robot-irl
    python3 scripts/export_preference_features.py --root data/isabella_irl_60d_openai_v2

前置要求：
    - 已运行过：
        - scripts/export_fusion_features.py
        - scripts/export_temporal_embeddings.py
    - 即 <root>/features 下已经存在：
        - fusion_daily.npy
        - rolling_stats.npy
        - window_valence.npy
        - X_daily.npy
        - feature_names.json

输出：
    - <root>/features/preference_features.npy
    - <root>/features/preference_feature_names.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.preference_features import export_preference_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export weekly preference features for IRL V2.")
    parser.add_argument(
        "--root",
        type=str,
        default="data/isabella_irl_60d_openai_v2",
        help="Dataset root directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_preference_features(args.root)


if __name__ == "__main__":
    main()

