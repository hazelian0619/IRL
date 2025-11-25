"""
RobotInterviewer
================

Phase 2 需要一个可重复触发的定点对话脚本，按照 L1-L6 访谈层级引导
机器人与 Alice（或其他 agent）对话。本模块通过 OpenAI/ OpenRouter
兼容接口驱动 LLM，同时内置摘要器生成多模态采集所需的元数据。
"""

from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from openai import OpenAI
from story.isabella_story import phase_for_day


CONVERSATION_FLOW: Sequence[Dict[str, str]] = [
    {"stage": "L1_opening", "prompt": "嗨，今天过得怎么样？"},
    {"stage": "L2_events", "prompt": "今天发生了什么特别的事吗？"},
    {"stage": "L3_emotion", "prompt": "这些事情让你有什么感受？"},
    {"stage": "L4_reflection", "prompt": "今天最让你印象深刻的时刻是什么？"},
    {"stage": "L5_planning", "prompt": "明天有什么计划或者期待吗？"},
    {
        "stage": "L6_scoring",
        "prompt": "如果给今天的整体心情打分(1-10)，请用“Score: <数字>. Explanation: …”的格式回答。",
    },
]


def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name) or default


def _default_headers() -> Dict[str, str]:
    headers = {}
    referer = _env("OPENROUTER_HTTP_REFERER")
    title = _env("OPENROUTER_APP_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def load_agent_profile(agent_name: str) -> Dict:
    """复用 persona 文件生成 agent 档案。"""
    personas_dir = Path(__file__).parent.parent / "data" / "personas"
    bio_path = personas_dir / f"{agent_name}_biography_prompt.txt"
    preset_path = personas_dir / "preset_personality.json"
    if not bio_path.exists():
        raise FileNotFoundError(f"Biography prompt not found: {bio_path}")
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset personality json not found: {preset_path}")

    biography = bio_path.read_text(encoding="utf-8")
    import json

    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    big_five = {
        dim: preset["big_five_parameters"][dim]["value"] for dim in ["O", "C", "E", "A", "N"]
    }

    return {
        "name": agent_name.title(),
        "biography": biography,
        "preset_scores": big_five,
    }


@dataclass
class ConversationTurn:
    speaker: str
    text: str
    stage: str

    def to_markdown(self, idx: int) -> str:
        role = "🤖 Robot" if self.speaker == "robot" else f"🧑 {self.speaker}"
        return f"{idx}. **{role}** ({self.stage}): {self.text}"


class RobotInterviewer:
    """实现定点对话逻辑，并产出多模态摘要。"""

    def __init__(
        self,
        agent_profile: Dict,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        self.agent_profile = agent_profile
        # 默认使用环境变量中的模型名；如果未设置，则退回到 gpt-3.5-turbo
        self.model = model or _env("TOWN_OPENROUTER_MODEL", "gpt-3.5-turbo")
        base_url = base_url or _env("OPENROUTER_API_BASE") or _env("OPENAI_BASE_URL")
        api_key = api_key or _env("OPENROUTER_API_KEY") or _env("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY/OPENAI_API_KEY 未配置，无法运行对话脚本。")
        self.client = OpenAI(api_key=api_key, base_url=base_url, default_headers=_default_headers())
        # 当前对话对应的 day_index（由 run_session 设置），用于为 Prompt 提供阶段信息。
        self._current_day_index: int | None = None

    def run_session(
        self,
        day_index: int,
        scheduled_time: datetime,
        conversation_flow: Sequence[Dict[str, str]] | None = None,
    ) -> Dict:
        """运行完整会话，返回 transcript + 摘要。"""
        # 记录当前 day_index，便于 _generate_agent_reply 使用阶段信息。
        self._current_day_index = day_index
        flow = conversation_flow or CONVERSATION_FLOW
        history: List[ConversationTurn] = []
        for item in flow:
            question = item["prompt"]
            history.append(ConversationTurn("robot", question, item["stage"]))
            reply = self._generate_agent_reply(question, history, scheduled_time)
            history.append(ConversationTurn(self.agent_profile["name"], reply, item["stage"]))

        transcript_md = self._render_transcript(history)
        summary = self._summarize_session(history)
        result = {
            "day": day_index,
            "scheduled_time": scheduled_time.isoformat(),
            "turns": len(history),
            "transcript_md": transcript_md,
            "mood_score": summary.get("mood_score", 5),
            "behaviors": summary.get("behaviors", []),
            "emotion": {
                "label": summary.get("dominant_emotion", "neutral"),
                "intensity": summary.get("tone_intensity", 0.5),
                "reason": summary.get("emotion_reason", ""),
            },
            "metadata": summary,
        }
        return result

    def _render_transcript(self, history: List[ConversationTurn]) -> str:
        lines = [turn.to_markdown(idx + 1) for idx, turn in enumerate(history)]
        return "\n".join(lines)

    def _generate_agent_reply(
        self,
        question: str,
        history: List[ConversationTurn],
        scheduled_time: datetime,
    ) -> str:
        # 基于当前 day_index 推断故事阶段，为 Prompt 提供世界演化的线索。
        phase_text = ""
        if self._current_day_index is not None:
            try:
                phase = phase_for_day(self._current_day_index)
                phase_text = textwrap.dedent(
                    f"""
                    今天是观察期的第 {self._current_day_index} 天，你当前所处的生活阶段是：{phase['name']}。
                    阶段简介：{phase['summary']}
                    你可以根据这个阶段的氛围，回顾今天在 Hobbs Cafe、小镇街道、公寓等场景中的真实经历。
                    """
                )
            except Exception:
                phase_text = ""

        # 固定人格与世界设定（基础框架）
        persona_desc = textwrap.dedent(
            f"""
            你是 {self.agent_profile['name']}。以下是你的人格简介（简写）：
            {self.agent_profile['biography'][:800]}...
            当前时间：{scheduled_time.strftime('%Y-%m-%d %H:%M')}，地点：斯坦福小镇 the Ville。
            {phase_text}

            你是生活在小镇中的真实人，而不是聊天机器人。你可以感到开心、满足，也可以感到疲惫、
            焦虑、失落或复杂，不需要每一天都报喜不报忧。

            回答问题时，请：
            - 结合你今天在小镇中的真实经历（例如在 Hobbs Cafe、合住公寓、学校等地的互动）；
            - 同时说出好的部分和不好的部分，而不是只讲“一切都很好”；
            - 如果今天真的很累、很烦或很一般，也可以坦诚地说出来。
            """
        )

        # 最近几轮对话片段，帮助模型保持上下文
        convo_snippet = "\n".join(
            f"{turn.speaker}: {turn.text}" for turn in history[-6:]
        )
        prompt = f"""{persona_desc}
继续以第一人称和机器人对话。保持自然、具体、细节化，参考“今天”在小镇中的生活经历。
最近对话：
{convo_snippet}

机器人提问：{question}
"""
        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                temperature=0.7,
                max_output_tokens=300,
            )
            text = getattr(response, "output_text", None)
            if text:
                return text.strip()
            # 如果接口返回了空文本，则认为调用失败，抛出异常由上层处理。
            raise RuntimeError("LLM 调用返回空响应")
        except Exception as exc:
            # 不再返回“看起来正常”的占位文本，而是抛出异常，让上层感知失败并中止当日采集。
            raise RuntimeError(f"LLM 调用失败: {exc}") from exc

    def _summarize_session(self, history: List[ConversationTurn]) -> Dict:
        convo_text = "\n".join(f"{t.speaker}: {t.text}" for t in history)
        summary_prompt = f"""
你是一个认真做日记总结的心理观察者。下面是一段“机器人”和人物之间的对话，请你根据对话内容，
给出这一天的真实摘要，而不是粉饰太平。

请阅读以下对话记录，并生成一个 JSON 对象，格式如下：
{{
  "behaviors": [
    {{"time": "今天早上/今天下午/今天晚上/全天/其他具体时间", "description": "用一两句话描述发生的关键事件"}},
    ...
  ],
  "dominant_emotion": "...",   # 从 ["开心","放松","累","焦虑","失落","生气","平静","复杂"] 中选一个最接近的
  "emotion_reason": "...",
  "tone_intensity": 0.0-1.0,
  "mood_score": <1-10数字>
}}
对话：
{convo_text}

关于 mood_score，请严格遵守：
- 1–3：整体偏糟糕，压力大、失落或负面体验明显；
- 4–6：有好有坏，比较一般、复杂或者有明显疲惫/担忧；
- 7–8：整体不错，有累但也有满足感；
- 9–10：非常好，非常满足和开心，负面情绪很少。

如果文本中出现 "Score: <数字>" 格式，请优先把该数字作为 mood_score；
如果没有出现 Score，请根据对话内容做出真实判断，不要为了“显得正面”而总是给 8–9 分。

要求：
- 至少给出 3 个 behaviors，尽量覆盖早上/下午/晚上或不同时间段；
- dominant_emotion 必须来自上面给出的列表；
- mood_score 必须是 1 到 10 之间的整数。
"""
        try:
            response = self.client.responses.create(
                model=self.model,
                input=summary_prompt,
                temperature=0.2,
                max_output_tokens=400,
            )
            raw = getattr(response, "output_text", "{}")
            import json

            cleaned = raw.strip()
            if not cleaned.startswith("{"):
                cleaned = cleaned[cleaned.find("{") :]
            summary = json.loads(cleaned)
        except Exception:
            # 如果解析失败，不直接静默吞掉错误，而是返回一个显式的“未知”摘要，
            # 方便后续判断哪些天缺少情绪标签。
            summary = {
                "behaviors": [],
                "dominant_emotion": "unknown",
                "emotion_reason": "",
                "tone_intensity": 0.0,
            }

        if "mood_score" not in summary:
            summary["mood_score"] = self._infer_score_from_history(history)
        return summary

    def _infer_score_from_history(self, history: List[ConversationTurn]) -> float:
        mood = 5.0
        score_regex = re.compile(r"score\s*[:：]\s*(\d+)", re.IGNORECASE)
        for turn in history:
            if turn.speaker == self.agent_profile["name"]:
                match = score_regex.search(turn.text)
                if match:
                    mood = float(match.group(1))
        return mood
