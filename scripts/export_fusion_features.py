#!/usr/bin/env python3
"""
Export multi-modal emotion fusion results for a 60-day dataset.

Usage:
    cd companion-robot-irl
    python3 scripts/export_fusion_features.py --root data/isabella_irl_3d_clean

This will:
    - run the rule-based single-modality predictors (text/behavior/emotion/score);
    - compute per-modality accuracies vs GT labels (from emotions/day_xxx.json);
    - derive fusion weights via softmax over accuracies;
    - save:
        - text_probs.npy
        - behavior_probs.npy
        - emotion_probs.npy
        - score_probs.npy
        - fusion_daily.npy
        - fusion_meta.json

These artifacts live under <root>/features and will be consumed by
the temporal feature extractor and sequence encoder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.emotion_fusion import compute_fusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export multi-modal fusion probabilities.")
    parser.add_argument(
        "--root",
        type=str,
        default="data/isabella_irl_3d_clean",
        help="Dataset root (contains conversations/behaviors/emotions/scores).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compute_fusion(args.root)
    print(f"[INFO] Fusion computed for dataset {args.root}")
    print(f"[INFO] Days: {result.days[0]}..{result.days[-1]} (total {len(result.days)})")
    print("[INFO] Per-modality accuracies:")
    for name, acc in result.accuracies.items():
        print(f"  - {name:8s}: {acc:.3f}")
    print("[INFO] Fusion weights (softmax over accuracies):")
    for name, w in result.weights.items():
        print(f"  - {name:8s}: {w:.3f}")


if __name__ == "__main__":
    main()

