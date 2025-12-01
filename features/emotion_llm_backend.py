"""
LLM-based emotion backend for daily text/behavior modalities.

设计目的：
- 为文本与行为两个模态提供统一的情绪分类后端；
- 返回与 CANONICAL_LABELS 对齐的 4 维概率分布，用于多模态融合；
- 在环境未配置 LLM 时自动退回到规则化实现（由调用方处理）。

说明：
- 我们复用 external_town 中已经存在的 LLM 封装 `call_llama`，以统一接入
  OpenRouter / 本地 Llama / 兼容 OpenAI 的网关；
- 通过环境变量 `IRL_FUSION_USE_LLM_BACKEND` 控制是否启用该后端：
    - 未设置或为 \"0\"/\"false\" → 不启用（返回 None，由上层 fallback）；
    - 为 \"1\"/\"true\"/\"on\" → 尝试调用 LLM；若失败则返回 None。
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

try:
    # 复用 Town 环境里已经实现好的 LLM 调用封装。
    from external_town.reverie.backend_server.persona.prompt_template.gpt_structure import (
        call_llama,
    )

    _LLM_BACKEND_AVAILABLE = True
except Exception:
    call_llama = None
    _LLM_BACKEND_AVAILABLE = False

# 与 emotion_fusion.CANONICAL_LABELS 保持一致的顺序。
CANONICAL_LABELS = ("积极", "消极", "中性", "复杂")
LABEL_INDEX: Dict[str, int] = {lbl: i for i, lbl in enumerate(CANONICAL_LABELS)}


def is_llm_backend_enabled() -> bool:
    """
    是否启用 LLM backend 的开关。

    只有在：
        - external_town 的 call_llama 可用；
        - 环境变量 IRL_FUSION_USE_LLM_BACKEND ∈ {\"1\",\"true\",\"on\"}
    时返回 True。
    """
    if not _LLM_BACKEND_AVAILABLE:
        return False
    flag = os.getenv("IRL_FUSION_USE_LLM_BACKEND", "").strip().lower()
    return flag in ("1", "true", "on", "yes")


def _empty_probs() -> np.ndarray:
    """返回 None 时由上层 fallback；这里提供一个占位 helper。"""
    return np.zeros(len(CANONICAL_LABELS), dtype=np.float32)


def _parse_emotion_json(raw: str) -> Optional[np.ndarray]:
    """
    解析 LLM 返回的 JSON 字符串，期望形式类似：

    {
      "积极": 0.7,
      "消极": 0.1,
      "中性": 0.1,
      "复杂": 0.1
    }

    若解析失败或数值异常，返回 None。
    """
    if not raw:
        return None
    try:
        # 尝试截取第一个 '{' 到最后一个 '}'，避免前后有多余文本。
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = raw[start : end + 1]
        obj = json.loads(snippet)
    except Exception:
        return None

    vec = np.zeros(len(CANONICAL_LABELS), dtype=np.float32)
    for lbl, idx in LABEL_INDEX.items():
        val = obj.get(lbl)
        if isinstance(val, (int, float)):
            vec[idx] = max(0.0, float(val))
    s = float(vec.sum())
    if not np.isfinite(s) or s <= 0.0:
        return None
    vec /= s
    return vec.astype(np.float32)


def llm_emotion_probs_from_text(text: str) -> Optional[np.ndarray]:
    """
    使用统一的 LLM backend 对一段中文文本做日级情绪分类。

    返回：
        - 4 维概率向量（积极/消极/中性/复杂），若解析失败则返回 None。
    """
    if not is_llm_backend_enabled():
        return None
    if not text:
        return None
    if call_llama is None:
        return None

    prompt = (
        "你是一个情绪分析助手。请阅读下面的中文文本，判断这一整天的整体情绪倾向，"
        "并给出四类情绪的概率分布。四类情绪为：\"积极\"、\"消极\"、\"中性\"、\"复杂\"。\\n"
        "请严格按照以下要求输出：\\n"
        "1. 输出一个 JSON 对象，不要输出其他解释性文字；\\n"
        "2. JSON 必须包含这四个键：\"积极\"、\"消极\"、\"中性\"、\"复杂\"；\\n"
        "3. 每个键的值是 0 到 1 之间的数字，四个值的和约等于 1。\\n"
        "下面是文本：\\n"
        "```\\n"
        f"{text}\\n"
        "```\\n"
        "现在请输出 JSON："
    )
    try:
        raw = call_llama(prompt, temperature=0.0)
    except Exception:
        return None
    return _parse_emotion_json(str(raw))


def _behaviors_to_text(behaviors: List[Dict]) -> str:
    """
    将一天内的行为列表转换为一段适合 LLM 理解的文本摘要。
    """
    parts: List[str] = []
    for beh in behaviors or []:
        time = str(beh.get("time", "") or "")
        desc = str(beh.get("description", "") or "")
        if time and desc:
            parts.append(f"{time}: {desc}")
        elif desc:
            parts.append(desc)
    return "\n".join(parts).strip()


def llm_emotion_probs_from_behaviors(behaviors: List[Dict]) -> Optional[np.ndarray]:
    """
    使用同一套 LLM backend，从“行为视角”判断这一整天的情绪倾向。

    输入：
        behaviors: DailyRecord.behaviors 列表。

    返回：
        - 4 维概率向量（积极/消极/中性/复杂），若解析失败则返回 None。
    """
    if not is_llm_backend_enabled():
        return None
    if call_llama is None:
        return None

    beh_text = _behaviors_to_text(behaviors)
    if not beh_text:
        return None

    prompt = (
        "你是一个情绪分析助手。下面是某一天的行为记录，请只从行为本身出发，判断这一天的整体情绪倾向，"
        "并给出四类情绪的概率分布。四类情绪为：\"积极\"、\"消极\"、\"中性\"、\"复杂\"。\\n"
        "请严格按照以下要求输出：\\n"
        "1. 输出一个 JSON 对象，不要输出其他解释性文字；\\n"
        "2. JSON 必须包含这四个键：\"积极\"、\"消极\"、\"中性\"、\"复杂\"；\\n"
        "3. 每个键的值是 0 到 1 之间的数字，四个值的和约等于 1。\\n"
        "下面是行为记录：\\n"
        "```\\n"
        f"{beh_text}\\n"
        "```\\n"
        "现在请输出 JSON："
    )
    try:
        raw = call_llama(prompt, temperature=0.0)
    except Exception:
        return None
    return _parse_emotion_json(str(raw))

