#!/usr/bin/env python3
"""
run_isabella_bfi_post.py
========================

用途：
  在已经完成 60 天 IRL 之后，对 Isabella 再做一次 BFI-44 “后测”，
  与你之前的前测
    validation/Isabella_Rodriguez_pretest_REAL_LLM_*.json
  保持同一套问卷与 LLM 流程。

实现方式：
  - 复用 agents/bfi_interviewer.py 里的 BFIInterviewer；
  - 仅将 report['test_type'] 从 'pretest_REAL' 改为 'posttest_REAL'；
  - 报告仍然保存在 validation/ 目录下。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.bfi_interviewer import BFIInterviewer  # noqa: E402


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BFI-44 posttest for Isabella (REAL LLM by default)."
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="Isabella Rodriguez",
        help="Agent full name.",
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

    interviewer = BFIInterviewer()

    use_llm = args.method == "llm"

    print("============================================================")
    print("BFI-44 Posttest Configuration")
    print("============================================================")
    print(f"Agent : {args.agent}")
    print(f"Method: {'Real LLM' if use_llm else 'Fallback (rule-based)'}")
    print()

    report = interviewer.complete_questionnaire(args.agent, use_llm=use_llm)

    # 覆盖 test_type，标记为 posttest_REAL，方便后续分析区分前后测。
    if use_llm:
        report["test_type"] = "posttest_REAL"
    else:
        report["test_type"] = "posttest_FALLBACK"

    report_path = interviewer.save_report(report)

    print("============================================================")
    print("BFI-44 Posttest Complete")
    print("============================================================")
    print(f"Agent: {report['agent_name']}")
    print(f"Method: {report['method']}")
    print(f"Questions answered: {report['total_questions']}")
    print(f"Report: {report_path}")
    print()


if __name__ == "__main__":
    main()

