"""
Multi-modal emotion fusion for 60-day IRL datasets.

设计目标（对齐《12》的 Step 4）：
- 为文本 / 行为 / 表情 / 分数这四个模态分别给出日级情绪预测；
- 以 `emotions/day_xxx.json['label']` 作为初始 GT，计算每个模态的精度；
- 根据精度得到权重向量 w；
- 做 late fusion，得到 60 天的融合情绪概率序列 `fusion_daily.npy`；
- 再从融合概率中导出一个 1D valence 序列，供后续滑窗与 LSTM 使用。

说明：
- 这里的单模态预测器是规则化 V0 实现，重点是在结构上对齐多模态融合的 pipeline；
- 后续可以在不改变接口的前提下，把内部模型替换为更强的文本/行为分类器。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils.dataset_loader import DailyDataset
from features.emotion_llm_backend import (
    llm_emotion_probs_from_text,
    llm_emotion_probs_from_behaviors,
)


CANONICAL_LABELS: Tuple[str, ...] = ("积极", "消极", "中性", "复杂")
_LABEL_INDEX: Dict[str, int] = {lbl: i for i, lbl in enumerate(CANONICAL_LABELS)}


def canonical_label(raw: str) -> str:
    """Map free-form emotion label strings into a small canonical set."""
    if not raw:
        return "中性"
    text = str(raw).strip()

    # 优先识别“复杂”
    if "复杂" in text:
        return "复杂"

    # 积极相关
    for w in ("开心", "高兴", "满足", "幸福", "兴奋", "感激", "温暖", "期待", "积极"):
        if w in text:
            return "积极"

    # 消极相关
    for w in ("难过", "沮丧", "失落", "压力", "紧张", "焦虑", "烦躁", "生气", "愤怒", "崩溃", "糟糕", "消极"):
        if w in text:
            return "消极"

    # 如明确标为“中性”
    if "中性" in text:
        return "中性"

    # 其他未识别标签统一归入“复杂”，以便后续人工审核。
    return "复杂"


def label_to_onehot(label: str) -> np.ndarray:
    idx = _LABEL_INDEX.get(canonical_label(label), _LABEL_INDEX["中性"])
    vec = np.zeros(len(CANONICAL_LABELS), dtype=np.float32)
    vec[idx] = 1.0
    return vec


def _text_probs(day_features: Dict[str, float]) -> np.ndarray:
    """
    文本模态情绪预测（V0 规则版）。

    说明：
    - 若启用了 LLM backend，则在 compute_fusion 中优先调用
      `llm_emotion_probs_from_text`，本函数作为 fallback 使用；
    - 这里保留最初的关键词计数逻辑，确保在无 LLM 环境下仍可运行。
    """
    pos = day_features.get("pos_word_count", 0.0)
    neg = day_features.get("neg_word_count", 0.0)
    if pos == 0 and neg == 0:
        return label_to_onehot("中性")
    if pos > neg:
        return label_to_onehot("积极")
    if neg > pos:
        return label_to_onehot("消极")
    return label_to_onehot("复杂")


def _behavior_probs(day_features: Dict[str, float]) -> np.ndarray:
    """
    行为模态情绪预测（V0 规则版）。

    规则（启发式）：
        - 高 social + 有 rest → 积极；
        - 高 work + 少 rest + mood_score 低 → 消极；
        - rest 多但 social/work 少 → 中性/恢复；
        - 其他 → 复杂。
    """
    social = day_features.get("beh_social_count", 0.0)
    work = day_features.get("beh_work_count", 0.0)
    rest = day_features.get("beh_rest_count", 0.0)
    mood = day_features.get("mood_score", 5.0)

    if social >= 1 and rest >= 1 and mood >= 6:
        return label_to_onehot("积极")
    if work >= 2 and rest == 0 and mood <= 4:
        return label_to_onehot("消极")
    if rest >= 1 and social == 0 and work == 0:
        return label_to_onehot("中性")
    return label_to_onehot("复杂")


def _emotion_probs(raw_label: str) -> np.ndarray:
    """直接使用 emotions JSON 提供的 label 作为情绪预测。"""
    return label_to_onehot(raw_label)


def _score_probs(mood_score: float) -> np.ndarray:
    """
    基于 1-10 mood_score 的粗略情绪预测：
        - 1-3  → 消极
        - 4-6  → 中性/复杂
        - 7-10 → 积极
    """
    if mood_score <= 3:
        return label_to_onehot("消极")
    if mood_score >= 7:
        return label_to_onehot("积极")
    # 中间区间区分“中性”和“复杂”的边界可以后续精化，这里先统一归为“复杂”。
    return label_to_onehot("复杂")


@dataclass
class FusionResult:
    days: List[int]
    gt_labels: List[str]
    probs_text: np.ndarray  # (T, K)
    probs_beh: np.ndarray
    probs_emotion: np.ndarray
    probs_score: np.ndarray
    fusion_probs: np.ndarray  # (T, K)
    weights: Dict[str, float]
    accuracies: Dict[str, float]


def compute_fusion(dataset_root: str | Path) -> FusionResult:
    """
    计算多模态情绪预测与加权融合结果。

    返回 FusionResult，并将各模态概率与融合结果保存到 <root>/features。
    """
    root = Path(dataset_root)
    ds = DailyDataset(root)
    feat_dir = root / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    days = ds.days
    gt_labels: List[str] = []
    probs_text: List[np.ndarray] = []
    probs_beh: List[np.ndarray] = []
    probs_em: List[np.ndarray] = []
    probs_score: List[np.ndarray] = []

    for day in days:
        rec = ds.get_day(day)
        gt = canonical_label(rec.emotion.get("label", ""))
        gt_labels.append(gt)

        # 文本模态：必须使用 LLM backend，若未启用或失败则报错。
        llm_text = llm_emotion_probs_from_text(rec.transcript_md)
        if llm_text is None:
            raise RuntimeError(
                "LLM 文本情绪 backend 返回 None；"
                "请确认已 source openai.env.local 且设置 IRL_FUSION_USE_LLM_BACKEND=true，"
                "并检查 external_town 的 call_llama 与 JSON 输出格式。"
            )
        probs_text.append(llm_text)

        # 行为模态：同样只接受 LLM backend。
        llm_beh = llm_emotion_probs_from_behaviors(rec.behaviors)
        if llm_beh is None:
            raise RuntimeError(
                "LLM 行为情绪 backend 返回 None；"
                "请确认已 source openai.env.local 且设置 IRL_FUSION_USE_LLM_BACKEND=true，"
                "并检查 external_town 的 call_llama 与 JSON 输出格式。"
            )
        probs_beh.append(llm_beh)

        probs_em.append(_emotion_probs(rec.emotion.get("label", "")))
        probs_score.append(_score_probs(rec.mood_score))

    probs_text = np.stack(probs_text, axis=0)
    probs_beh = np.stack(probs_beh, axis=0)
    probs_em = np.stack(probs_em, axis=0)
    probs_score = np.stack(probs_score, axis=0)

    # 计算每个模态的精度（argmax vs GT）。
    def acc(probs: np.ndarray) -> float:
        preds = probs.argmax(axis=1)
        gt_idx = np.array([_LABEL_INDEX[g] for g in gt_labels], dtype=np.int64)
        return float((preds == gt_idx).mean())

    acc_text = acc(probs_text)
    acc_beh = acc(probs_beh)
    acc_em = acc(probs_em)
    acc_score = acc(probs_score)

    accuracies = {
        "text": acc_text,
        "behavior": acc_beh,
        "emotion": acc_em,
        "score": acc_score,
    }

    # 使用 softmax(acc) 计算权重，避免出现负权重。
    acc_vec = np.array(list(accuracies.values()), dtype=np.float32)
    # 为了数值稳定和区分度，对 acc 做一个简单缩放。
    logits = acc_vec / max(acc_vec.max(), 1e-6)
    w = np.exp(logits)
    w = w / w.sum()
    weights = {name: float(w[i]) for i, name in enumerate(accuracies.keys())}

    # 融合：加权求和各模态概率。
    fusion_probs = (
        weights["text"] * probs_text
        + weights["behavior"] * probs_beh
        + weights["emotion"] * probs_em
        + weights["score"] * probs_score
    )

    # 保存到 features 目录。
    np.save(feat_dir / "text_probs.npy", probs_text)
    np.save(feat_dir / "behavior_probs.npy", probs_beh)
    np.save(feat_dir / "emotion_probs.npy", probs_em)
    np.save(feat_dir / "score_probs.npy", probs_score)
    np.save(feat_dir / "fusion_daily.npy", fusion_probs)

    meta = {
        "days": days,
        "canonical_labels": CANONICAL_LABELS,
        "accuracies": accuracies,
        "weights": weights,
    }
    (feat_dir / "fusion_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return FusionResult(
        days=days,
        gt_labels=gt_labels,
        probs_text=probs_text,
        probs_beh=probs_beh,
        probs_emotion=probs_em,
        probs_score=probs_score,
        fusion_probs=fusion_probs,
        weights=weights,
        accuracies=accuracies,
    )


def fusion_valence(fusion_probs: np.ndarray) -> np.ndarray:
    """
    从融合概率中导出 1D valence 序列：
        valence = P(积极) - P(消极)
    """
    if fusion_probs.ndim != 2 or fusion_probs.shape[1] != len(CANONICAL_LABELS):
        raise ValueError(f"Unexpected fusion_probs shape: {fusion_probs.shape}")
    idx_pos = _LABEL_INDEX["积极"]
    idx_neg = _LABEL_INDEX["消极"]
    return (fusion_probs[:, idx_pos] - fusion_probs[:, idx_neg]).astype(np.float32)
