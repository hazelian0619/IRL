#!/usr/bin/env python3
"""
run_town_irl_days.py
====================

按“12”的思路，把小镇世界运行 + nightly 访谈绑在一起：

- 从一个已有的 3-agent 小镇存档 fork 出一个新模拟（默认：July1_the_ville_isabella_maria_klaus-step-3-20）
- 用 Reverie 的后端每次推进“1 天”的世界
- 每天结束后，用我们自己的 RobotInterviewer 对 Isabella 做一次 L1–L6 访谈
- 把 nightly 结果写入 Phase 2 数据结构：
    conversations/day_001.md
    behaviors/day_001.json
    emotions/day_001.json
    scores/mood_scores.csv

注意：
- 这里不会调用 Reverie 内部的 _run_daily_checkin（避免重复访谈和额外 token）。
- 目前只针对 Isabella（persona_name="Isabella Rodriguez"，agent="isabella"）。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# IRL 访谈相关
from agents.robot_interviewer import RobotInterviewer, load_agent_profile
from utils.data_collector import DataCollector
from story.isabella_story import IsabellaStoryState

# Reverie 后端（小镇环境）
BACKEND_ROOT = ROOT / "external_town" / "reverie" / "backend_server"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from reverie import ReverieServer  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run town simulation + nightly IRL interviews for Isabella."
    )
    parser.add_argument(
        "--fork-sim",
        type=str,
        default="July1_the_ville_isabella_maria_klaus-step-3-20",
        help="作为起点的 existing simulation 名称（storage 下的文件夹名）。",
    )
    parser.add_argument(
        "--sim-code",
        type=str,
        default="isabella_n3_irl_3d",
        help="本次实验的新 simulation 名称（会在 storage 下创建同名目录）。",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="要额外模拟的天数（每一天结束后做一次 nightly 访谈）。",
    )
    parser.add_argument(
        "--sec-per-step",
        type=int,
        default=86400,
        help=(
            "每一步对应的游戏秒数。默认 86400 = 一步 = 一天，"
            "可以改成 43200（半天两步）等以调节速度和粒度。"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/isabella_irl_from_town",
        help="Phase 2 nightly 数据输出目录。",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="可选：覆盖默认 LLM 模型名（否则用环境变量 TOWN_OPENROUTER_MODEL）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 确保 Reverie 使用我们期望的时间步长。
    os.environ["TOWN_SEC_PER_STEP"] = str(args.sec_per_step)

    # 为 nightly 数据输出目录构造绝对路径，避免后续切换工作目录导致相对路径错位。
    output_root_path = (ROOT / args.output_root).resolve()

    # IMPORTANT:
    # Reverie 使用的是相对路径 "../../environment/frontend_server/storage"，
    # 这个相对路径是以 backend_server 目录为基准设计的。
    # 所以这里临时把工作目录切到 BACKEND_ROOT，再构造 ReverieServer。
    prev_cwd = os.getcwd()
    os.chdir(BACKEND_ROOT)
    try:
        # 1. 启动一个新的小镇模拟（从 fork-sim 拷贝一份出来）
        rs = ReverieServer(args.fork_sim, args.sim_code)

        # 打印当前小镇里加载的 personas，确保确实是 3 个 agent（而不是 n25）。
        try:
            persona_names = list(getattr(rs, "personas", {}).keys())
        except Exception:
            persona_names = []
        print(f"[Town] Loaded personas ({len(persona_names)}): {persona_names}")

        # 计算“每天需要多少步”——与 reverie.py 里的 auto_run_days 保持一致。
        sec_per_step = getattr(rs, "sec_per_step", args.sec_per_step)
        steps_per_day = max(1, int(86400 // sec_per_step))
        print(
            f"[Town] sec_per_step={sec_per_step}, steps_per_day={steps_per_day}, "
            f"days={args.days}"
        )

        # 2. 准备我们的 nightly 访谈器 + 数据写入器
        profile = load_agent_profile("isabella")
        interviewer = RobotInterviewer(profile, model=args.model)
        collector = DataCollector(str(output_root_path))
        story_state = IsabellaStoryState()

        # 说明：fork 的存档里已经包含了一段历史；我们只关心“新跑出来的这几天”。
        # 每次循环：
        #   - 先让世界前进 steps_per_day 步（大步长，每步=全天/半天）
        #   - 再用小镇时间的“今天”做一次 nightly 访谈
        for day_index in range(1, args.days + 1):
            # 让世界多过一天（或半天等）
            print(f"[Town] Running simulated day {day_index} ...")
            rs.start_server(steps_per_day)
            rs.save()

            # 当前 curr_time 是“刚迈入明天”的时间点；
            # 我们把访谈日期设为“刚刚结束的这一天”的晚上 20:00。
            sim_day_date = (rs.curr_time - timedelta(days=1)).date()
            scheduled_dt = datetime.combine(sim_day_date, time(hour=20, minute=0))

            print(
                f"[IRL] Nightly interview for Isabella, "
                f"sim_date={sim_day_date.isoformat()}, day_index={day_index}"
            )
            result = interviewer.run_session(day_index=day_index, scheduled_time=scheduled_dt)

            # 用 IsabellaStoryState 调整 mood_score，让 60 天曲线更符合阶段 & 文本。
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
                "sim_code": args.sim_code,
            }
            collector.save_conversation(day_index, result["transcript_md"], metadata)
            collector.save_behaviors(day_index, result["behaviors"])
            collector.save_emotions(day_index, {"day": day_index, **result["emotion"]})
            collector.append_mood_score(day_index, result["mood_score"])

            print(
                f"[IRL] Day {day_index} done | "
                f"mood={result['mood_score']} | emotion={result['emotion']['label']}"
            )

        print("=======================================")
        print("Town + Nightly IRL run completed.")
        print("=======================================")
        print(f"- Simulation code : {args.sim_code}")
        print(f"- Nightly data dir: {output_root_path}")
    finally:
        # 恢复原来的工作目录，避免影响后续其它脚本。
        os.chdir(prev_cwd)


if __name__ == "__main__":
    main()
