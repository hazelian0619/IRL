"""
Temporal feature extraction for 60-day IRL datasets.

Implements the sliding-window statistics and exponential-decay baseline
described in the planning doc (`进度/12`):

- 60-day fused daily labels/scores -> 7-day sliding window stats:
    (54 windows, each with mean/var/max/min/trend)
    -> saved as /features/rolling_stats.npy
- Exponentially weighted baseline of the full 60-day sequence:
    -> saved as /features/global_baseline.npy
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from utils.dataset_loader import DailyDataset
from features.emotion_fusion import fusion_valence


@dataclass
class SlidingWindowConfig:
    window: int = 7
    step: int = 1


@dataclass
class DecayConfig:
    lambda_: float = 0.05  # decay hyper-parameter from the design doc


def _sliding_windows(seq: np.ndarray, window: int, step: int) -> np.ndarray:
    """
    Generate sliding windows over a 1D sequence.

    Args:
        seq: shape (T,)
        window: window size
        step: stride between windows

    Returns:
        array of shape (num_windows, window)
    """
    if seq.ndim != 1:
        raise ValueError(f"Expected 1D sequence, got shape {seq.shape}")
    T = seq.shape[0]
    if T < window:
        raise ValueError(f"Sequence too short for window={window}: length={T}")
    windows = []
    for start in range(0, T - window + 1, step):
        windows.append(seq[start : start + window])
    return np.stack(windows, axis=0).astype(np.float32)


def compute_window_stats(seq: np.ndarray, cfg: SlidingWindowConfig) -> np.ndarray:
    """
    Compute 7-day sliding window statistics for a 1D daily sequence.

    For each window we compute:
        - mean
        - variance
        - max
        - min
        - linear trend (slope)

    Returns:
        stats: shape (num_windows, 5)
    """
    windows = _sliding_windows(seq, cfg.window, cfg.step)  # (N, W)
    N, W = windows.shape

    mean = windows.mean(axis=1)
    var = windows.var(axis=1)
    max_ = windows.max(axis=1)
    min_ = windows.min(axis=1)

    # trend via simple linear regression slope on index 0..W-1 for each window
    t = np.arange(W, dtype=np.float32)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum()
    if t_var == 0:
        raise ValueError("Unexpected zero variance in time indices for trend computation")
    # center time indices once
    t_centered = t - t_mean
    slopes = np.empty(N, dtype=np.float32)
    for i in range(N):
        y = windows[i]
        y_mean = y.mean()
        cov = float(((t_centered) * (y - y_mean)).sum())
        slopes[i] = cov / t_var

    stats = np.stack([mean, var, max_, min_, slopes], axis=1)
    return stats.astype(np.float32)


def compute_exponential_baseline(seq: np.ndarray, cfg: DecayConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute an exponentially weighted baseline over the full sequence.

    We follow an EWMA-style scheme:
        - weights_t = lambda * (1 - lambda)^(T-1-t)
        - normalized over t=0..T-1

    Args:
        seq: shape (T,)
        cfg: decay config

    Returns:
        baseline: shape (1,)  - scalar baseline for the sequence
        weights: shape (T,)   - normalized weights used
    """
    if seq.ndim != 1:
        raise ValueError(f"Expected 1D sequence, got shape {seq.shape}")
    T = seq.shape[0]
    lam = float(cfg.lambda_)
    if not (0.0 < lam < 1.0):
        raise ValueError(f"lambda_ must be in (0,1), got {lam}")

    # Recent days should have higher weight.
    idx = np.arange(T, dtype=np.float32)
    weights = lam * np.power(1.0 - lam, T - 1 - idx)
    weights = weights / weights.sum()
    baseline = float((weights * seq).sum())
    return np.array([baseline], dtype=np.float32), weights.astype(np.float32)


def export_temporal_features(dataset_root: str | Path) -> None:
    """
    Convenience helper: read a 60-day dataset and export temporal features
    to <root>/features:

        - rolling_stats.npy: (num_windows=54, 5) matrix
        - global_baseline.npy: (1,) scalar baseline
        - decay_weights.npy: (T,) daily weights
        - temporal_meta.json: config and shapes

    Currently we operate on a 1D daily emotion/valence sequence:
        - if fusion_daily.npy exists, use fusion_valence (P(积极)-P(消极));
        - otherwise, fall back to raw mood_score.
    """
    root = Path(dataset_root)
    ds = DailyDataset(root)
    out_dir = root / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1D 序列：优先使用融合概率导出的 valence；若不存在，则退回到 mood_score。
    fusion_path = out_dir / "fusion_daily.npy"
    if fusion_path.exists():
        fusion_probs = np.load(fusion_path)  # (T, K)
        y = fusion_valence(fusion_probs)
        series_name = "fusion_valence"
    else:
        y = np.array([ds.get_day(d).mood_score for d in ds.days], dtype=np.float32)
        series_name = "mood_score"

    sw_cfg = SlidingWindowConfig()
    decay_cfg = DecayConfig()

    rolling_stats = compute_window_stats(y, sw_cfg)
    baseline, weights = compute_exponential_baseline(y, decay_cfg)

    np.save(out_dir / "rolling_stats.npy", rolling_stats)
    np.save(out_dir / "global_baseline.npy", baseline)
    np.save(out_dir / "decay_weights.npy", weights)

    meta = {
        "dataset_root": str(root),
        "days": ds.days,
        "sequence_length": len(ds.days),
        "sliding_window": {
            "window": sw_cfg.window,
            "step": sw_cfg.step,
            "num_windows": int(rolling_stats.shape[0]),
            "num_stats": int(rolling_stats.shape[1]),
        },
        "decay": {
            "lambda": decay_cfg.lambda_,
        },
        "source_series": series_name,
        "outputs": {
            "rolling_stats": str(out_dir / "rolling_stats.npy"),
            "global_baseline": str(out_dir / "global_baseline.npy"),
            "decay_weights": str(out_dir / "decay_weights.npy"),
        },
    }
    (out_dir / "temporal_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
