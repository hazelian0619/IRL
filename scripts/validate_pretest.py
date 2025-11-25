"""
验证BFI-44前测结果 - 计算与预设参数的相关性
"""

import json
import sys
from pathlib import Path

# 添加agents目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'agents'))

from bfi_validator import BFI44Validator


def validate_pretest_report(report_path: str):
    """验证前测报告"""

    print(f"{'='*60}")
    print(f"BFI-44 Pretest Validation")
    print(f"{'='*60}\n")

    # 加载报告
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    print(f"Agent: {report['agent_name']}")
    print(f"Test Date: {report['test_date']}")
    print(f"Method: {report['method']}")
    print(f"Total Questions: {report['total_questions']}\n")

    # 提取回答分数
    responses = report['responses_scores']
    responses_int = {int(k): int(v) for k, v in responses.items()}

    # 提取预设参数
    preset_scores = report['agent_profile']['preset_scores']

    print(f"Preset Parameters:")
    for dim, score in preset_scores.items():
        print(f"  {dim}: {score:.2f}")
    print()

    # 创建验证器
    validator = BFI44Validator()

    # 计算测量得分
    print("Calculating dimension scores from responses...")
    measured_scores = validator.calculate_dimension_scores(responses_int)

    print(f"\nMeasured Scores:")
    for dim, score in measured_scores.items():
        print(f"  {dim}: {score:.4f}")
    print()

    # 验证相关性
    print("Validating against preset...")
    validation = validator.validate_against_preset(measured_scores, preset_scores)

    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}\n")

    print(f"Pearson Correlation: r = {validation['pearson_r']:.4f}")
    print(f"P-value: {validation['p_value']:.4f}")
    print(f"Mean Absolute Error: {validation['mae']:.4f}")
    print(f"Threshold: r > {validation['threshold']}")
    print(f"\nStatus: {'✅ PASSED' if validation['passed'] else '❌ FAILED'}\n")

    print(f"Dimension-wise Comparison:")
    print(f"{'Dim':<5} {'Preset':<10} {'Measured':<10} {'Error':<10}")
    print(f"{'-'*40}")
    for dim in ['O', 'C', 'E', 'A', 'N']:
        preset_val = validation['preset_scores'][dim]
        measured_val = validation['measured_scores'][dim]
        error = validation['dimension_errors'][dim]
        print(f"{dim:<5} {preset_val:<10.4f} {measured_val:<10.4f} {error:<10.4f}")

    print(f"\n{'='*60}")

    if validation['passed']:
        print("✅ BFI-44 Pretest PASSED!")
        print("Alice's personality is consistent with preset parameters.")
        print("\nNext Steps:")
        print("  1. Lock Ground Truth parameters")
        print("  2. Proceed to Step 2.1: Daily interaction system")
        print("  3. Begin 60-day data collection")
    else:
        print("❌ BFI-44 Pretest FAILED")
        print(f"Correlation r={validation['pearson_r']:.4f} is below threshold ({validation['threshold']})")
        print("\nRecommended Actions:")
        print("  1. Adjust alice_biography_prompt.txt")
        print("  2. Re-inject Alice with updated prompt")
        print("  3. Repeat BFI-44 pretest")

    print(f"\n{'='*60}\n")

    return validation


if __name__ == "__main__":
    # 找到最新的pretest报告
    validation_dir = Path(__file__).parent.parent / "validation"

    # 查找所有pretest_REAL文件
    pretest_files = list(validation_dir.glob("*_pretest_REAL_*.json"))

    if not pretest_files:
        print("ERROR: No pretest_REAL report found in validation/")
        print("Please run: python agents/bfi_interviewer.py")
        sys.exit(1)

    # 使用最新的文件
    latest_report = max(pretest_files, key=lambda p: p.stat().st_mtime)

    print(f"Using report: {latest_report.name}\n")

    # 执行验证
    validation = validate_pretest_report(latest_report)

    # 保存验证结果
    output_path = latest_report.parent / f"validation_{latest_report.stem}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(validation, f, indent=2, ensure_ascii=False)

    print(f"Validation result saved: {output_path}")
