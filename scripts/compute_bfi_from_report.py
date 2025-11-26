#!/usr/bin/env python3
"""
Compute Big Five scores from a stored BFI-44 report.

用法示例：

    cd companion-robot-irl
    python3 scripts/compute_bfi_from_report.py \\
        --report validation/Isabella_Rodriguez_pretest_REAL_LLM_20251126_164921.json

脚本会：
    - 读取指定的 BFI-44 报告（由 BFIInterviewer 生成）；
    - 从 responses_scores 中恢复 44 题的 1-5 分数；
    - 使用 BFI44Validator.calculate_dimension_scores 计算 O/C/E/A/N 五维 0-1 得分；
    - 可选：与 preset_personality.json 中的预设人格做一次相关性验证；
    - 输出到 validation/validation_<原文件名>.json。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.bfi_validator import BFI44Validator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Big Five scores from BFI-44 report JSON.")
    parser.add_argument(
        "--report",
        type=str,
        required=True,
        help="Path to BFI-44 report JSON (e.g. validation/Isabella_...pretest_REAL_*.json).",
    )
    parser.add_argument(
        "--no-preset-validation",
        action="store_true",
        help="If set, skip validation against preset_personality.json and only output measured scores.",
    )
    return parser.parse_args()


def load_responses_from_report(report: Dict) -> Dict[int, int]:
    """
    Extract {question_id: score} mapping from a BFI report.
    """
    if "responses_scores" in report:
        raw = report["responses_scores"]
        # keys may be strings; normalize to int
        return {int(k): int(v) for k, v in raw.items()}

    # Fallback: build from responses_detailed
    resp = {}
    for item in report.get("responses_detailed", []):
        qid = int(item["question_id"])
        resp[qid] = int(item["score"])
    return resp


def main() -> None:
    args = parse_args()
    report_path = Path(args.report)
    if not report_path.exists():
        raise SystemExit(f"Report not found: {report_path}")

    raw = json.loads(report_path.read_text(encoding="utf-8"))
    agent_name = raw.get("agent_name", "UNKNOWN")
    test_type = raw.get("test_type", "UNKNOWN")

    validator = BFI44Validator()
    responses = load_responses_from_report(raw)
    measured_scores = validator.calculate_dimension_scores(responses)

    result = {
        "agent_name": agent_name,
        "test_type": test_type,
        "source_report": str(report_path),
        "measured_scores": measured_scores,
    }

    if not args.no_preset_validation:
        # 加载预设人格（当前 Alice/Isabella 共用同一份 preset_personality.json）
        preset_path = Path("data/personas/preset_personality.json")
        if preset_path.exists():
            preset = json.loads(preset_path.read_text(encoding="utf-8"))
            preset_scores = {
                dim: preset["big_five_parameters"][dim]["value"]
                for dim in ["O", "C", "E", "A", "N"]
            }
            val = validator.validate_against_preset(measured_scores, preset_scores)
            result["validation"] = val
        else:
            result["validation"] = {"warning": "preset_personality.json not found"}

    out_name = "validation_" + report_path.name
    out_path = report_path.parent / out_name
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] Computed Big Five scores for {agent_name}")
    print(f"[INFO] Measured scores: {measured_scores}")
    if "validation" in result:
        val = result["validation"]
        if "pearson_r" in val:
            print(
                f"[INFO] Validation vs preset: r={val['pearson_r']:.4f}, "
                f"MAE={val['mae']:.4f}, passed={val['passed']}"
            )
    print(f"[INFO] Saved validation report to {out_path}")


if __name__ == "__main__":
    main()
