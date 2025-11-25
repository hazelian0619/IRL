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

_BEHAVIOR_CATEGORIES = {
    # 社交相关：与人互动、对话、聚会等
    "social": (
        "和朋友",
        "与朋友",
        "与人",
        "聊天",
        "交谈",
        "对话",
        "见面",
        "约会",
        "聚会",
        "派对",
        "客人",
        "顾客",
        "社区",
    ),
    # 工作 / 任务：准备、布置、工作、开会等
    "work": (
        "工作",
        "上班",
        "准备",
        "布置",
        "打工",
        "任务",
        "项目",
        "排班",
        "值班",
        "忙碌",
    ),
    # 休息 / 自我照顾：睡觉、休息、放松等
    "rest": (
        "休息",
        "睡觉",
        "小憩",
        "放松",
        "散步",
        "走走",
        "喝咖啡",
        "喝茶",
        "放空",
    ),
    # 创作 / 艺术 / 爱好
    "creative": (
        "画画",
        "绘画",
        "艺术",
        "展览",
        "作品",
        "写作",
        "写日记",
        "音乐",
        "练习",
    ),
}


def _count_keywords(text: str, keywords: Tuple[str, ...]) -> int:
    if not text:
        return 0
    count = 0
    for w in keywords:
        count += len(re.findall(re.escape(w), text))
    return count


def _classify_behavior(description: str) -> Dict[str, int]:
    """
    Classify a behavior description into coarse categories based on keyword rules.

    One description can contribute to multiple categories (e.g. social + work).
    """
    desc = description or ""
    counts: Dict[str, int] = {k: 0 for k in _BEHAVIOR_CATEGORIES.keys()}
    for cat, words in _BEHAVIOR_CATEGORIES.items():
        for w in words:
            if w in desc:
                counts[cat] += 1
    return counts


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

    # Coarse behavior categories (social / work / rest / creative).
    beh_cat_counts: Dict[str, float] = {f"beh_{k}_count": 0.0 for k in _BEHAVIOR_CATEGORIES.keys()}
    for beh in record.behaviors:
        desc = str(beh.get("description", "") or "")
        cat_counts = _classify_behavior(desc)
        for cat, cnt in cat_counts.items():
            beh_cat_counts[f"beh_{cat}_count"] += float(cnt > 0)
    feats.update(beh_cat_counts)

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
