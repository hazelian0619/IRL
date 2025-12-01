#!/usr/bin/env python3
"""
Experiment: calibrate a scalar valence mapping from fused emotion probabilities
against self-reported mood scores, and compare it with the current
hand-crafted valence = P(积极) - P(消极) baseline.

Usage:
    cd companion-robot-irl
    python3 scripts/experiment_valence_calibration.py --root data/isabella_irl_3d_clean

This script does NOT modify any existing pipeline artifacts. It only:
    - reads <root>/features/fusion_daily.npy (P_fusion(day));
    - reads mood scores via DailyDataset;
    - fits a 4-parameter linear mapping:
          ms_norm(day) ~= w_pos*p_pos + w_neg*p_neg + w_neu*p_neu + w_comp*p_comp
      where ms_norm is a simple normalization of mood_score into roughly [-1, 1];
    - compares correlation / MSE between:
          valence_raw(day) = P(积极) - P(消极)
          valence_cal(day) = w · P_fusion(day)
      and the normalized mood scores.

The goal is to provide a quantitative check of whether a calibrated mapping
gives a closer match to mood scores than the naive P_pos - P_neg definition.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dataset_loader import DailyDataset
from features.emotion_fusion import fusion_valence, CANONICAL_LABELS  # type: ignore


def _load_fusion_probs(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load fused probabilities and associated day indices from <root>/features.
    """
    feat_dir = root / "features"
    fusion_path = feat_dir / "fusion_daily.npy"
    meta_path = feat_dir / "fusion_meta.json"

    if not fusion_path.exists():
        raise FileNotFoundError(f"Missing fusion_daily.npy under {feat_dir}")
    fusion_probs = np.load(fusion_path)  # (T, 4)

    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        days = np.array(meta.get("days", list(range(1, fusion_probs.shape[0] + 1))), dtype=int)
    else:
        # Fallback: assume consecutive days starting at 1.
        days = np.arange(1, fusion_probs.shape[0] + 1, dtype=int)

    if fusion_probs.ndim != 2 or fusion_probs.shape[1] != len(CANONICAL_LABELS):
        raise ValueError(f"Unexpected fusion_probs shape: {fusion_probs.shape}")

    return fusion_probs.astype(np.float32), days


def _load_mood_scores(root: Path, days: np.ndarray) -> np.ndarray:
    """
    Load mood scores via DailyDataset and align them with the given day indices.
    """
    ds = DailyDataset(root)
    ds_days = np.array(ds.days, dtype=int)
    if not np.array_equal(ds_days, days):
        # Be strict here so we do not silently misalign scores.
        raise ValueError(f"Day indices mismatch: fusion_meta days={days.tolist()} dataset days={ds_days.tolist()}")
    scores = np.array([ds.get_day(int(d)).mood_score for d in days], dtype=np.float32)
    return scores


def _normalize_mood(scores: np.ndarray) -> np.ndarray:
    """
    Map mood scores from approximately [1,10] into roughly [-1,1].

    The exact constants are not critical; this is a simple affine rescaling
    that keeps the interpretation "low score → negative valence, high score → positive valence".
    """
    return (scores - 5.5) / 4.5


def _fit_linear_mapping(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fit a minimal linear mapping y ≈ X @ w using least squares.

    X: (T, 4) fused probabilities
    y: (T,) normalized mood scores

    Returns:
        w: (4,) weight vector.
    """
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"Expected X shape (T,4), got {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(f"Shape mismatch: X={X.shape}, y={y.shape}")

    # Solve min ||Xw - y||^2 via least squares.
    w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return w.astype(np.float32)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim != 1 or b.ndim != 1 or a.shape[0] != b.shape[0]:
        raise ValueError(f"Correlation expects 1D arrays of same length, got {a.shape}, {b.shape}")
    if a.shape[0] < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_experiment(dataset_root: str | Path) -> None:
    root = Path(dataset_root)

    fusion_probs, days = _load_fusion_probs(root)  # (T,4), (T,)
    mood = _load_mood_scores(root, days)  # (T,)
    mood_norm = _normalize_mood(mood)  # (T,)

    # Baseline: naive valence = P(积极) - P(消极).
    val_raw = fusion_valence(fusion_probs)  # (T,)

    # Calibrated: valence_cal = w · P_fusion(day), with w fitted against mood_norm.
    X = fusion_probs  # (T,4)
    w = _fit_linear_mapping(X, mood_norm)  # (4,)
    val_cal = X @ w  # (T,)

    # Metrics: correlation & MSE vs normalized mood.
    corr_raw = _corr(val_raw, mood_norm)
    corr_cal = _corr(val_cal, mood_norm)
    mse_raw = float(np.mean((val_raw - mood_norm) ** 2))
    mse_cal = float(np.mean((val_cal - mood_norm) ** 2))

    print(f"[INFO] Dataset root       : {root}")
    print(f"[INFO] Num days           : {fusion_probs.shape[0]}")
    print(f"[INFO] Learned weights w  :")
    for lbl, coeff in zip(CANONICAL_LABELS, w):
        print(f"    w[{lbl}] = {coeff:+.4f}")
    print()
    print("[INFO] Alignment with mood_score (normalized to ~[-1,1]]):")
    print(f"    corr(mood_norm, valence_raw) = {corr_raw:+.4f}")
    print(f"    corr(mood_norm, val_calibrated) = {corr_cal:+.4f}")
    print(f"    MSE (valence_raw  vs mood_norm) = {mse_raw:.4f}")
    print(f"    MSE (val_calibrated vs mood_norm) = {mse_cal:.4f}")

    # Optional: print a small sample of days for sanity.
    print()
    print("[INFO] Sample days (day, mood_score, mood_norm, val_raw, val_calibrated):")
    for i in range(min(5, fusion_probs.shape[0])):
        print(
            f"    day {int(days[i]):3d}: "
            f"mood={mood[i]:4.1f}, "
            f"m_norm={mood_norm[i]:+5.2f}, "
            f"v_raw={val_raw[i]:+5.2f}, "
            f"v_cal={val_cal[i]:+5.2f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valence calibration experiment.")
    parser.add_argument(
        "--root",
        type=str,
        default="data/isabella_irl_3d_clean",
        help="Dataset root (must already have fusion_daily.npy under features/).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(args.root)


if __name__ == "__main__":
    main()

