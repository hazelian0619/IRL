"""
Dataset loading utilities for 60-day IRL experiments.

This module provides a standardized way to read multi-modal daily data from
folders like ``data/isabella_irl_3d_clean``:

    <root>/
        conversations/day_001.md
        behaviors/day_001.json
        emotions/day_001.json
        scores/mood_scores.csv

The goal is to make the data access pattern identical for:
    - current synthetic 60-day datasets; and
    - future REAL datasets ingested from the Town environment or human users.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass
class DailyRecord:
    """Container for all modalities of a single day."""

    day: int
    agent: str
    date: Optional[datetime]
    turns: Optional[int]
    transcript_md: str
    behaviors: List[Dict]
    emotion: Dict
    mood_score: float


class DailyDataset:
    """
    Multi-modal 60-day dataset loader.

    This class assumes the canonical directory layout used by
    ``data/isabella_irl_3d_clean`` and ``data/alice_irl_60d``.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.conv_dir = self.root / "conversations"
        self.behavior_dir = self.root / "behaviors"
        self.emotion_dir = self.root / "emotions"
        self.scores_path = self.root / "scores" / "mood_scores.csv"

        for path in (self.conv_dir, self.behavior_dir, self.emotion_dir, self.scores_path):
            if not path.exists():
                raise FileNotFoundError(f"Missing expected path in dataset: {path}")

        self._scores, self._score_duplicates = self._load_scores(self.scores_path)
        self._days = sorted(self._scores.keys())

    @staticmethod
    def _load_scores(path: Path) -> tuple[Dict[int, float], List[int]]:
        scores: Dict[int, float] = {}
        duplicates: List[int] = []
        with path.open("r", encoding="utf-8") as fp:
            reader = csv.reader(fp)
            header = next(reader, None)
            if header is None or len(header) < 2 or header[0].strip() != "day":
                raise ValueError(f"Invalid mood_scores header in {path}")
            for row in reader:
                if not row:
                    continue
                try:
                    day = int(row[0])
                    score = float(row[1])
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"Invalid row in mood_scores: {row}") from exc
                if day in scores:
                    # Keep the latest entry but track duplicates for reporting.
                    duplicates.append(day)
                scores[day] = score
        if not scores:
            raise ValueError(f"No scores found in {path}")
        return scores, duplicates

    @property
    def days(self) -> List[int]:
        """Sorted list of day indices present in the dataset."""
        return list(self._days)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._days)

    def __iter__(self) -> Iterator[DailyRecord]:
        for day in self._days:
            yield self.get_day(day)

    # ---- internal helpers -------------------------------------------------

    def _format_day(self, day: int) -> str:
        return f"day_{day:03d}"

    def _load_conversation(self, day: int) -> Dict:
        tag = self._format_day(day)
        path = self.conv_dir / f"{tag}.md"
        if not path.exists():
            raise FileNotFoundError(f"Conversation file missing for day {day}: {path}")
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if len(lines) < 5 or not lines[0].startswith("# Conversation"):
            raise ValueError(f"Conversation header malformed in {path}")

        agent = "unknown"
        date: Optional[datetime] = None
        turns: Optional[int] = None

        for ln in lines[1:4]:
            if ln.startswith("- agent:"):
                agent = ln.split(":", 1)[1].strip()
            elif ln.startswith("- date:"):
                raw = ln.split(":", 1)[1].strip()
                try:
                    date = datetime.fromisoformat(raw)
                except Exception:
                    date = None
            elif ln.startswith("- turns:"):
                try:
                    turns = int(ln.split(":", 1)[1].strip())
                except Exception:
                    turns = None

        # Transcript starts after the metadata header and a blank line.
        try:
            blank_idx = lines.index("")
        except ValueError:
            blank_idx = 4
        transcript = "\n".join(lines[blank_idx + 1 :]).strip()

        return {
            "agent": agent,
            "date": date,
            "turns": turns,
            "transcript_md": transcript,
        }

    def _load_json_file(self, directory: Path, day: int, expected_key: str) -> Dict:
        tag = self._format_day(day)
        path = directory / f"{tag}.json"
        if not path.exists():
            raise FileNotFoundError(f"JSON file missing for day {day}: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON parse error in {path}: {exc}") from exc

        if "day" in data and int(data["day"]) != day:
            raise ValueError(f"Day mismatch in {path}: header={data['day']} expected={day}")
        if expected_key and expected_key not in data:
            raise ValueError(f"Missing key '{expected_key}' in {path}")
        return data

    # ---- public API -------------------------------------------------------

    def get_day(self, day: int) -> DailyRecord:
        """
        Load all modalities for a given day.

        Raises FileNotFoundError / ValueError when any modality is missing or malformed.
        """
        if day not in self._scores:
            raise KeyError(f"Day {day} not present in mood_scores at {self.scores_path}")

        conv_meta = self._load_conversation(day)
        beh = self._load_json_file(self.behavior_dir, day, expected_key="behaviors")
        emo = self._load_json_file(self.emotion_dir, day, expected_key="label")

        return DailyRecord(
            day=day,
            agent=conv_meta["agent"],
            date=conv_meta["date"],
            turns=conv_meta["turns"],
            transcript_md=conv_meta["transcript_md"],
            behaviors=list(beh.get("behaviors", [])),
            emotion=emo,
            mood_score=self._scores[day],
        )

    def validate(self) -> List[str]:
        """
        Run a series of sanity checks over the dataset.

        Returns:
            A list of warning strings; raises on hard failures.
        """
        warnings: List[str] = []

        # Check day indices are contiguous.
        expected = list(range(min(self._days), max(self._days) + 1))
        if self._days != expected:
            warnings.append(
                f"Non-contiguous days detected: found={self._days} expected={expected}"
            )

        # Report duplicate score entries if any.
        if getattr(self, "_score_duplicates", None):
            uniq = sorted(set(self._score_duplicates))
            warnings.append(
                f"Duplicate mood_scores entries for days: {uniq} "
                "(using the last occurrence for each)"
            )

        # Check that every day has all files and metadata is consistent.
        for day in self._days:
            record = self.get_day(day)
            if record.turns is None or record.turns <= 0:
                warnings.append(f"Day {day}: missing or invalid turn count")
            if not record.transcript_md:
                warnings.append(f"Day {day}: empty transcript")
            if not record.behaviors:
                warnings.append(f"Day {day}: empty behaviors list")
            if "label" not in record.emotion:
                warnings.append(f"Day {day}: emotion label missing")

        return warnings


def iter_datasets(root: str | Path) -> Iterable[DailyDataset]:
    """
    Discover dataset roots under ``data/`` and yield DailyDataset instances.

    This is a convenience helper when we want to run the same checks across
    multiple experimental folders (e.g. synthetic vs REAL).
    """
    base = Path(root)
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if not (child / "scores" / "mood_scores.csv").exists():
            continue
        yield DailyDataset(child)
