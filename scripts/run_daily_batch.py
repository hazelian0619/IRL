#!/usr/bin/env python3
"""
run_daily_batch.py
==================

用于快速生成多天 Phase 2 数据（方便补齐 60 天任务）。
示例：
    python scripts/run_daily_batch.py --agent alice --start-day 1 \
        --days 5 --start-date 2025-11-17 --output-root data/alice_experiment
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.robot_interviewer import RobotInterviewer, load_agent_profile
from utils.data_collector import DataCollector
from story.isabella_story import IsabellaStoryState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate daily conversations/data.")
    parser.add_argument("--agent", default="alice", help="Agent name.")
    parser.add_argument("--start-day", type=int, default=1, help="起始 day index（1-based）。")
    parser.add_argument("--days", type=int, default=5, help="生成天数。")
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="开始日期 (YYYY-MM-DD)。缺省则使用今天。",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/alice_experiment",
        help="数据写入目录。",
    )
    parser.add_argument("--model", type=str, default=None, help="覆盖默认模型名。")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如目标目录已存在，则先清空其中的 conversations/behaviors/emotions/scores。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_date = (
        datetime.strptime(args.start_date, "%Y-%m-%d").date()
        if args.start_date
        else datetime.today().date()
    )
    profile = load_agent_profile(args.agent)
    interviewer = RobotInterviewer(profile, model=args.model)
    # 如果指定了 overwrite，则先删除旧的数据子目录，避免多次运行时交叉污染。
    output_root = Path(args.output_root)
    if args.overwrite and output_root.exists():
        for sub in ("conversations", "behaviors", "emotions", "scores"):
            target = output_root / sub
            if target.exists():
                # 小心起见，只删除我们自己创建的子目录
                import shutil

                shutil.rmtree(target)
    collector = DataCollector(str(output_root))

    # 仅在 Isabella 实验中启用故事状态驱动的情绪模型。
    story_state: IsabellaStoryState | None = None
    if profile["name"].lower().startswith("isabella"):
        story_state = IsabellaStoryState()

    summary_rows = []
    for offset in range(args.days):
        day_index = args.start_day + offset
        scheduled_date = base_date + timedelta(days=offset)
        scheduled_dt = datetime.combine(scheduled_date, time(hour=20, minute=0))

        try:
            result = interviewer.run_session(day_index=day_index, scheduled_time=scheduled_dt)
        except Exception as exc:  # noqa: BLE001
            # 如果当日对话生成失败（通常是 LLM/网络问题），停止批量运行，
            # 避免写入“看起来正常但实际上是占位”的伪数据。
            print(f"[ERROR] Failed to generate session for day {day_index}: {exc}")
            print("       批量任务已中止，请检查网络/LLM 状态后从该天重新运行。")
            break

        # 如果启用了 IsabellaStoryState，则用它来调整 mood_score，使其更符合阶段与长期状态。
        if story_state is not None:
            story_state.day_index = day_index
            adjusted_score = story_state.compute_mood_score(
                transcript=result["transcript_md"],
                llm_score=result.get("mood_score"),
            )
            result["mood_score"] = adjusted_score

        metadata = {
            "agent": profile["name"],
            "date": scheduled_dt.isoformat(),
            "turns": result["turns"],
        }
        collector.save_conversation(day_index, result["transcript_md"], metadata)
        collector.save_behaviors(day_index, result["behaviors"])
        collector.save_emotions(day_index, {"day": day_index, **result["emotion"]})
        collector.append_mood_score(day_index, result["mood_score"])

        summary_rows.append(
            {
                "day": day_index,
                "date": scheduled_dt.strftime("%Y-%m-%d"),
                "turns": result["turns"],
                "mood": result["mood_score"],
                "emotion": result["emotion"]["label"],
            }
        )

    print("============================================================")
    print("Batch Daily Talks Completed")
    print("============================================================")
    for row in summary_rows:
        print(
            f"Day {row['day']:>3} | {row['date']} | turns={row['turns']:>2} | "
            f"mood={row['mood']:>2} | emotion={row['emotion']}"
        )
    print("数据已写入目录:", args.output_root)


if __name__ == "__main__":
    main()
