#!/usr/bin/env python3
"""
run_isabella_bfi_post_from_irl.py
=================================

用途：
  在完成 60 天 IRL 之后，让 Isabella 基于这 60 天的 nightly 日志
  回顾自己的生活，再回答一遍 BFI‑44 问卷，作为“后测”。

思路：
  - 从 IRL 数据目录（默认 data/isabella_irl_60d_openai_v2）中读取
    60 天的 nightly 对话；
  - 抽取每天 Isabella 的回答，压缩成一段“60 天生活概要”文本；
  - 在 BFIInterviewer 的每一道题 prompt 之前，加上这段概要，
    提醒她“此刻是在经历完 60 天之后作答”；
  - 将报告标记为 posttest_IRL_REAL，保存到 validation/ 目录。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.bfi_interviewer import BFIInterviewer  # noqa: E402


def build_irl_context(irl_root: Path, max_days: int = 60) -> str:
    """
    从 IRL 对话中提取每一天 Isabella 的“情绪回答”作为简短日记摘要。

    为了避免超过模型的上下文长度，这里只取每一天中 Isabella 在
    L3_emotion 阶段的那一条回答，并截断到约 120 个字符。
    """
    lines: list[str] = []
    conv_dir = irl_root / "conversations"
    for day in range(1, max_days + 1):
        conv_path = conv_dir / f"day_{day:03d}.md"
        if not conv_path.exists():
            continue
        text = conv_path.read_text(encoding="utf-8")

        day_emotion: str | None = None
        for ln in text.splitlines():
            # 只抓 L3_emotion 那一行，聚焦“今天整体感受”
            if "🧑 Isabella" in ln and "(L3_emotion)" in ln:
                parts = ln.split("):", 1)
                if len(parts) == 2:
                    content = parts[1].strip()
                else:
                    content = ln.strip()
                if content:
                    day_emotion = content
                    break

        if not day_emotion:
            continue

        # 严格截断到 ~120 个字符，避免 prompt 过长
        max_chars = 120
        if len(day_emotion) > max_chars:
            day_emotion = day_emotion[: max_chars - 3] + "..."
        lines.append(f"Day {day}: {day_emotion}")

    if not lines:
        return ""

    header = (
        "Below is a compressed diary of your last days in the town 'the Ville', "
        "based on nightly interviews (one emotional summary per day):\n"
    )
    return header + "\n".join(lines)


class IRLBFIInterviewer(BFIInterviewer):
    """在 BFI 提问 prompt 前注入 60 天 IRL 概要的版本。"""

    def __init__(self, extra_context: str, questionnaire_path: Optional[str] = None):
        super().__init__(questionnaire_path=questionnaire_path)
        self.extra_context = extra_context or ""

    def construct_prompt(self, agent_profile: dict, question: dict) -> str:  # type: ignore[override]
        base = super().construct_prompt(agent_profile, question)
        if not self.extra_context:
            return base
        prefix = f"""You are now answering this questionnaire AFTER having lived 60 days in the town "the Ville",
running Hobbs Cafe and interacting with other agents.

Please base your answers on who you are NOW, after these 60 days of experiences.

{self.extra_context}

"""
        return prefix + base


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BFI-44 posttest for Isabella, conditioned on 60-day IRL logs."
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="Isabella Rodriguez",
        help="Agent full name.",
    )
    parser.add_argument(
        "--irl-root",
        type=str,
        default="data/isabella_irl_60d_openai_v2",
        help="Root directory of 60-day IRL data (must contain conversations/).",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Number of IRL days to include in the summary.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["llm", "fallback"],
        default="llm",
        help="Use real LLM ('llm') or rule-based fallback ('fallback').",
    )
    return parser


def main() -> None:
    args = parse_args().parse_args()

    irl_root = (ROOT / args.irl_root).resolve()
    extra_context = build_irl_context(irl_root, max_days=args.days)

    interviewer = IRLBFIInterviewer(extra_context=extra_context)

    use_llm = args.method == "llm"

    print("============================================================")
    print("BFI-44 Posttest (IRL-conditioned) Configuration")
    print("============================================================")
    print(f"Agent    : {args.agent}")
    print(f"Method   : {'Real LLM' if use_llm else 'Fallback (rule-based)'}")
    print(f"IRL root : {irl_root}")
    print(f"IRL days : {args.days}")
    print()

    report = interviewer.complete_questionnaire(args.agent, use_llm=use_llm)

    # 标记为 IRL 后测，方便与 pretest 区分。
    if use_llm:
        report["test_type"] = "posttest_IRL_REAL"
    else:
        report["test_type"] = "posttest_IRL_FALLBACK"

    report_path = interviewer.save_report(report)

    print("============================================================")
    print("BFI-44 Posttest (IRL-conditioned) Complete")
    print("============================================================")
    print(f"Agent: {report['agent_name']}")
    print(f"Method: {report['method']}")
    print(f"Questions answered: {report['total_questions']}")
    print(f"Report: {report_path}")
    print()


if __name__ == "__main__":
    main()
