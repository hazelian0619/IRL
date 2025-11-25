"""
IsabellaStoryState
==================

为 Isabella Rodriguez 设计的 60 天情绪与故事状态模型（轻量版本）。

目标：
- 提供一个跨天的“故事阶段 + 情绪基线”框架；
- 根据每日对话文本中的正负事件信号，更新疲惫/压力等状态；
- 给出一个比单纯 LLM Score 更符合人格与生活起伏的 mood_score。

注意：
- 这是一个启发式模型，不宣称是唯一正确的心理模型；
- 但比“每天独立随口报一个 7 分”要合理得多。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Optional


# 60 天阶段划分（可以根据需要微调）
PHASES = [
    {
        "name": "valentine_prep",
        "start": 1,
        "end": 7,
        "summary": "情人节派对前的筹备期：白天在 Hobbs Cafe 忙着准备装饰、菜单和邀请，既期待又紧张。",
        "base_score": 7.5,
    },
    {
        "name": "valentine_peak",
        "start": 8,
        "end": 10,
        "summary": "情人节当天及紧接着的几天：派对现场热闹，顾客多，反馈好，但身体和精神都非常疲惫。",
        "base_score": 8.0,
    },
    {
        "name": "art_show",
        "start": 11,
        "end": 25,
        "summary": "艺术家展览与社区活动期：支持年轻艺术家、参与慈善活动，有成就感也有时间与精力上的拉扯。",
        "base_score": 7.0,
    },
    {
        "name": "election",
        "start": 26,
        "end": 40,
        "summary": "小镇选举期：Hobbs Cafe 成为政治讨论场，信息过载、争吵和价值观冲突带来额外压力。",
        "base_score": 6.5,
    },
    {
        "name": "recovery",
        "start": 41,
        "end": 60,
        "summary": "调整与重构期：主要活动结束，开始重新规划菜单、空间和自我节奏，有恢复也有空虚与反思。",
        "base_score": 7.0,
    },
]


def phase_for_day(day_index: int) -> Dict:
    """根据天数返回所属阶段定义。超过范围的天数落到最后一个阶段。"""
    for phase in PHASES:
        if phase["start"] <= day_index <= phase["end"]:
            return phase
    return PHASES[-1]


def _count_keywords(text: str, keywords: tuple[str, ...]) -> int:
    """统计文本中关键字的出现次数（简单启发式）。"""
    if not text:
        return 0
    return sum(text.count(k) for k in keywords)


POSITIVE_KEYWORDS = (
    "开心",
    "高兴",
    "满足",
    "幸福",
    "感激",
    "温暖",
    "被支持",
    "被理解",
    "成就感",
    "顺利",
    "放松",
    "轻松",
    "惊喜",
    "期待",
    "兴奋",
)

NEGATIVE_KEYWORDS = (
    "累",
    "疲惫",
    "疲劳",
    "压力",
    "紧张",
    "焦虑",
    "空虚",
    "空落",
    "烦躁",
    "生气",
    "吵闹",
    "争吵",
    "不舒服",
    "生病",
    "头疼",
    "感冒",
    "失落",
    "失望",
    "挫败",
    "心累",
)


@dataclass
class IsabellaStoryState:
    """跨天维护 Isabella 的简单情绪与压力状态。"""

    day_index: int = 1
    fatigue: float = 0.0  # 身体/精神疲惫
    stress: float = 0.0  # 压力、焦虑
    hope: float = 0.7  # 对未来的总体期待（0-1）

    def update_from_text(self, transcript: str) -> Dict[str, int]:
        """根据当天对话文本更新状态，并返回正负事件计数。"""
        pos = _count_keywords(transcript, POSITIVE_KEYWORDS)
        neg = _count_keywords(transcript, NEGATIVE_KEYWORDS)

        # 简单规则：负向词越多，fatigue / stress 越上涨；正向词略微对冲压力。
        if neg > 0:
            self.fatigue += 0.15 * neg
            self.stress += 0.2 * neg
        if pos > 0:
            self.hope += 0.05 * pos
            self.stress -= 0.05 * pos

        # 温和衰减，避免状态无限累加
        self.fatigue *= 0.9
        self.stress *= 0.9

        # 限制范围，避免极端值
        self.fatigue = max(0.0, min(self.fatigue, 3.0))
        self.stress = max(0.0, min(self.stress, 3.0))
        self.hope = max(0.0, min(self.hope, 1.0))

        return {"pos": pos, "neg": neg}

    def compute_mood_score(self, transcript: str, llm_score: Optional[float]) -> int:
        """
        根据当前阶段 + 状态 + 文本关键词 + LLM 的 score，给出一个 1-10 的 mood_score。

        设计思路：
        - 阶段 base_score 提供“这段时间平均水平”（比如选举期略低）；
        - 文本中的正负关键词和累积疲惫/压力调节当天的偏移；
        - LLM 的 score 作为一个温和的先验（不完全信任，避免天天 7）；
        - 最后加一点随机噪声，避免机械感。
        """
        stats = self.update_from_text(transcript or "")
        phase = phase_for_day(self.day_index)
        base = phase.get("base_score", 7.0)

        # 正负事件的即时影响（简单线性）
        pos_effect = 0.4 * stats["pos"]
        neg_effect = -0.6 * stats["neg"]

        # 累积疲惫与压力的影响
        fatigue_effect = -0.8 * self.fatigue
        stress_effect = -0.7 * self.stress

        score = base + pos_effect + neg_effect + fatigue_effect + stress_effect

        # 引入 LLM 的 score 作为温和的先验（避免完全无视它）
        if llm_score is not None:
            try:
                llm_score_f = float(llm_score)
                score = 0.5 * score + 0.5 * llm_score_f
            except Exception:
                pass

        # 加一点随机噪声，避免机械的重复值
        score += random.uniform(-0.5, 0.5)

        # 限制在 1-10 之间并四舍五入
        score = int(round(max(1.0, min(10.0, score))))
        return score

