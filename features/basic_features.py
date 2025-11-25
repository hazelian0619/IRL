"""
Basic daily feature extraction for 60-day IRL datasets.

This module builds lightweight, explainable features from the multi-modal
DailyRecord objects produced by ``utils.dataset_loader``. The goal is to have
an industrial-grade baseline that is:

- robust to missing fields;
- transparent and easy to debug;
- easy to extend with richer text/behavior embeddings later.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from utils.dataset_loader import DailyRecord


_POSITIVE_WORDS = (
    "开心",
    "高兴",
    "满足",
    "幸福",
    "轻松",
    "放松",
    "期待",
    "兴奋",
)
_NEGATIVE_WORDS = (
    "累",
    "疲惫",
    "压力",
    "紧张",
    "焦虑",
    "空虚",
    "烦躁",
    "失落",
    "挫败",
)


def _count_keywords(text: str, keywords: Tuple[str, ...]) -> int:
    if not text:
        return 0
    count = 0
    for w in keywords:
        count += len(re.findall(re.escape(w), text))
    return count


def extract_daily_features(record: DailyRecord) -> Dict[str, float]:
    """
    Extract a small, robust set of scalar features from a DailyRecord.

    These are deliberately simple and transparent:
        - conversation length (characters);
        - number of behavior snippets;
        - positive/negative sentiment keyword counts;
        - one-hot encoding of the emotion label (top few).
    """
    feats: Dict[str, float] = {}

    text = record.transcript_md or ""
    feats["len_chars"] = float(len(text))

    # Behavior statistics.
    feats["num_behaviors"] = float(len(record.behaviors))

    # Keyword-based rough sentiment.
    feats["pos_word_count"] = float(_count_keywords(text, _POSITIVE_WORDS))
    feats["neg_word_count"] = float(_count_keywords(text, _NEGATIVE_WORDS))

    # Emotion label one-hot (keep top common ones explicit, others in "other").
    label = str(record.emotion.get("label", "")).strip()
    known_labels = ("积极", "消极", "中性", "复杂")
    for lbl in known_labels:
        feats[f"emotion_{lbl}"] = 1.0 if label == lbl else 0.0
    if label and label not in known_labels:
        feats["emotion_other"] = 1.0
    else:
        feats["emotion_other"] = 0.0

    # Include mood score itself, which can later be re-used as target or feature.
    feats["mood_score"] = float(record.mood_score)

    return feats


def build_feature_matrix(records: List[DailyRecord]) -> Tuple[np.ndarray, List[str]]:
    """
    Convert a list of DailyRecord objects into a feature matrix.

    Returns:
        features: shape (num_days, num_features)
        feature_names: list of feature names aligned with columns
    """
    if not records:
        raise ValueError("No records provided to build_feature_matrix")

    # Collect features and ensure we have a stable column order.
    all_dicts: List[Dict[str, float]] = []
    all_keys: Counter = Counter()
    for rec in records:
        feats = extract_daily_features(rec)
        all_dicts.append(feats)
        all_keys.update(feats.keys())

    feature_names = sorted(all_keys.keys())
    mat = np.zeros((len(records), len(feature_names)), dtype=np.float32)
    for i, feats in enumerate(all_dicts):
        for j, name in enumerate(feature_names):
            mat[i, j] = float(feats.get(name, 0.0))

    return mat, feature_names

