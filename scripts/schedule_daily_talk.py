#!/usr/bin/env python3
"""
schedule_daily_talk.py
======================

Phase 2 对话调度脚本：
    python scripts/schedule_daily_talk.py --agent alice --day 1 \
        --output-root data/alice_experiment

执行后会：
1. 运行 RobotInterviewer，按照 L1-L6 流程生成对话记录；
2. 使用 DataCollector 写入 conversations/behaviors/emotions/scores；
3. 在终端打印本次会话概览，方便人工确认。
"""

from __future__ import annotations

import argparse
from datetime import datetime, time
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.robot_interviewer import RobotInterviewer, load_agent_profile
from utils.data_collector import DataCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schedule daily talk at 20:00.")
    parser.add_argument("--agent", default="alice", help="Agent name, e.g. alice/bob")
    parser.add_argument("--day", type=int, default=1, help="Day index (1-based).")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date string (YYYY-MM-DD). 默认今天。",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/alice_experiment",
        help="数据存储的根目录。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="覆盖默认的 LLM 模型名（可选）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scheduled_date = (
        datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else datetime.today().date()
    )
    scheduled_dt = datetime.combine(scheduled_date, time(hour=20, minute=0))

    profile = load_agent_profile(args.agent)
    interviewer = RobotInterviewer(profile, model=args.model)
    result = interviewer.run_session(day_index=args.day, scheduled_time=scheduled_dt)

    collector = DataCollector(args.output_root)
    metadata = {
        "agent": profile["name"],
        "date": scheduled_dt.isoformat(),
        "turns": result["turns"],
    }
    conv_path = collector.save_conversation(args.day, result["transcript_md"], metadata)
    beh_path = collector.save_behaviors(args.day, result["behaviors"])
    emo_path = collector.save_emotions(args.day, {"day": args.day, **result["emotion"]})
    score_path = collector.append_mood_score(args.day, result["mood_score"])

    print("============================================================")
    print("Daily Talk Completed")
    print("============================================================")
    print(f"Agent      : {profile['name']}")
    print(f"Day        : {args.day}")
    print(f"Scheduled  : {scheduled_dt}")
    print(f"Turns      : {result['turns']}")
    print(f"Mood score : {result['mood_score']}")
    print("Saved files:")
    print(f"  - Conversation : {conv_path}")
    print(f"  - Behaviors    : {beh_path}")
    print(f"  - Emotions     : {emo_path}")
    print(f"  - Mood scores  : {score_path}")


if __name__ == "__main__":
    main()
