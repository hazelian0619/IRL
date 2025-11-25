"""
DataCollector - 用于Phase 2多模态数据写入

负责把每日对话、行为、情绪、心情得分存入统一目录结构：
data/<experiment_root>/
    conversations/day_001.md
    behaviors/day_001.json
    emotions/day_001.json
    scores/mood_scores.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


class DataCollector:
    """处理Phase 2数据存储。"""

    def __init__(self, root: str):
        self.root = Path(root)
        self.conv_dir = self.root / "conversations"
        self.behavior_dir = self.root / "behaviors"
        self.emotion_dir = self.root / "emotions"
        self.score_file = self.root / "scores" / "mood_scores.csv"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for path in (
            self.root,
            self.conv_dir,
            self.behavior_dir,
            self.emotion_dir,
            self.score_file.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

        if not self.score_file.exists():
            with self.score_file.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow(["day", "score"])

    @staticmethod
    def _format_day(day_index: int) -> str:
        return f"day_{day_index:03d}"

    def save_conversation(self, day_index: int, transcript: str, metadata: Dict) -> Path:
        day_tag = self._format_day(day_index)
        path = self.conv_dir / f"{day_tag}.md"
        header_lines = [
            f"# Conversation {day_tag}",
            f"- agent: {metadata.get('agent', 'unknown')}",
            f"- date: {metadata.get('date', 'unknown')}",
            f"- turns: {metadata.get('turns', 0)}",
            "",
        ]
        with path.open("w", encoding="utf-8") as fp:
            fp.write("\n".join(header_lines))
            fp.write(transcript.strip() + "\n")
        return path

    def save_behaviors(self, day_index: int, behaviors: List[Dict]) -> Path:
        day_tag = self._format_day(day_index)
        path = self.behavior_dir / f"{day_tag}.json"
        with path.open("w", encoding="utf-8") as fp:
            json.dump({"day": day_index, "behaviors": behaviors}, fp, ensure_ascii=False, indent=2)
        return path

    def save_emotions(self, day_index: int, emotion_record: Dict) -> Path:
        day_tag = self._format_day(day_index)
        path = self.emotion_dir / f"{day_tag}.json"
        with path.open("w", encoding="utf-8") as fp:
            json.dump(emotion_record, fp, ensure_ascii=False, indent=2)
        return path

    def append_mood_score(self, day_index: int, score: float) -> Path:
        # 追加写入日级心情分数。
        with self.score_file.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow([day_index, score])
        return self.score_file
