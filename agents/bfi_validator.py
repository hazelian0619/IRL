"""
BFI-44施测与评分系统

功能：
1. 加载BFI-44问卷
2. 向Agent提问并收集答案
3. 计算五维得分
4. 与预设参数做相关性验证
5. 生成前测/后测报告

用法：
    python agents/bfi_validator.py --agent "Alice Chen" --test-type pretest
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.stats import pearsonr
from datetime import datetime


class BFI44Validator:
    """BFI-44人格问卷施测与验证"""

    def __init__(self, questionnaire_path: str = None):
        """初始化问卷"""
        if questionnaire_path is None:
            questionnaire_path = Path(__file__).parent.parent / "data" / "bfi44_questionnaire.json"

        with open(questionnaire_path, 'r', encoding='utf-8') as f:
            self.questionnaire = json.load(f)

        self.questions = self.questionnaire['questions']
        self.scoring = self.questionnaire['scoring_instructions']

    def get_questions(self) -> List[Dict]:
        """获取所有44题"""
        return self.questions

    def reverse_score(self, score: int) -> int:
        """反向计分"""
        reverse_map = {1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
        return reverse_map[score]

    def calculate_dimension_scores(self, responses: Dict[int, int]) -> Dict[str, float]:
        """
        计算五维得分

        Args:
            responses: {question_id: score(1-5)}

        Returns:
            {'O': 0.72, 'C': 0.61, 'E': 0.78, 'A': 0.67, 'N': 0.34}
        """
        dimension_scores = {}

        # 定义每个维度的题目
        dimensions = {
            'E': {'ids': [1,6,11,16,21,26,31,36], 'reverse': [6,21,31]},
            'A': {'ids': [2,7,12,17,22,27,32,37,42], 'reverse': [2,12,27,37]},
            'C': {'ids': [3,8,13,18,23,28,33,38,43], 'reverse': [8,18,23,43]},
            'N': {'ids': [4,9,14,19,24,29,34,39], 'reverse': [9,24,34]},
            'O': {'ids': [5,10,15,20,25,30,35,40,44], 'reverse': [35,41]}
        }

        for dim, config in dimensions.items():
            raw_scores = []

            for qid in config['ids']:
                if qid not in responses:
                    raise ValueError(f"Question {qid} not answered for dimension {dim}")

                score = responses[qid]

                # 反向计分
                if qid in config['reverse']:
                    score = self.reverse_score(score)

                raw_scores.append(score)

            # 归一化到0-1
            # 公式: (总分 - 题数×1) / (题数×4)
            total = sum(raw_scores)
            n_questions = len(config['ids'])
            normalized = (total - n_questions) / (n_questions * 4)

            dimension_scores[dim] = round(normalized, 4)

        return dimension_scores

    def validate_against_preset(self,
                                measured_scores: Dict[str, float],
                                preset_scores: Dict[str, float]) -> Dict:
        """
        验证测量值与预设值的一致性

        Args:
            measured_scores: 测量得到的5维分数
            preset_scores: 预设的5维分数

        Returns:
            validation_result包含相关系数、各维度误差等
        """
        # 确保维度顺序一致
        dimensions = ['O', 'C', 'E', 'A', 'N']

        measured_vector = [measured_scores[d] for d in dimensions]
        preset_vector = [preset_scores[d] for d in dimensions]

        # 计算Pearson相关系数
        r, p_value = pearsonr(measured_vector, preset_vector)

        # 计算各维度误差
        errors = {d: abs(measured_scores[d] - preset_scores[d]) for d in dimensions}
        mae = np.mean(list(errors.values()))

        # 判断是否通过
        passed = r > 0.75  # 文档line 221标准

        result = {
            'pearson_r': float(round(r, 4)),
            'p_value': float(round(p_value, 4)),
            'mae': float(round(mae, 4)),
            'dimension_errors': {k: float(v) for k, v in errors.items()},
            'passed': bool(passed),
            'threshold': 0.75,
            'measured_scores': measured_scores,
            'preset_scores': preset_scores
        }

        return result

    def generate_report(self,
                       agent_name: str,
                       test_type: str,
                       responses: Dict[int, int],
                       preset_scores: Dict[str, float] = None,
                       output_path: str = None) -> str:
        """
        生成测试报告

        Args:
            agent_name: Agent名称
            test_type: 'pretest' or 'posttest'
            responses: 问卷回答
            preset_scores: 预设参数（前测需要）
            output_path: 报告保存路径

        Returns:
            报告文件路径
        """
        # 计算得分
        measured_scores = self.calculate_dimension_scores(responses)

        # 创建报告
        report = {
            'agent_name': agent_name,
            'test_type': test_type,
            'test_date': datetime.now().isoformat(),
            'total_questions': 44,
            'responses': responses,
            'measured_scores': measured_scores
        }

        # 如果有预设参数，做验证
        if preset_scores:
            validation = self.validate_against_preset(measured_scores, preset_scores)
            report['validation'] = validation

        # 保存报告
        if output_path is None:
            output_dir = Path(__file__).parent.parent / "validation"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{agent_name.replace(' ', '_')}_{test_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def print_validation_summary(self, report_path: str):
        """打印验证摘要"""
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)

        print(f"\n{'='*60}")
        print(f"BFI-44 {report['test_type'].upper()} 验证报告")
        print(f"{'='*60}")
        print(f"Agent: {report['agent_name']}")
        print(f"测试时间: {report['test_date']}")
        print(f"\n【测量得分】")
        for dim, score in report['measured_scores'].items():
            print(f"  {dim}: {score:.4f}")

        if 'validation' in report:
            val = report['validation']
            print(f"\n【验证结果】")
            print(f"  Pearson相关系数 r: {val['pearson_r']:.4f}")
            print(f"  p-value: {val['p_value']:.4f}")
            print(f"  平均绝对误差 MAE: {val['mae']:.4f}")
            print(f"  阈值标准: r > {val['threshold']}")
            print(f"  验证状态: {'✅ 通过' if val['passed'] else '❌ 未通过'}")

            print(f"\n【各维度误差】")
            for dim, error in val['dimension_errors'].items():
                preset_val = val['preset_scores'][dim]
                measured_val = val['measured_scores'][dim]
                print(f"  {dim}: 预设={preset_val:.2f}, 测量={measured_val:.4f}, 误差={error:.4f}")

        print(f"\n{'='*60}")


def simulate_agent_responses(preset_scores: Dict[str, float],
                             noise_level: float = 0.1) -> Dict[int, int]:
    """
    模拟Agent回答（用于测试）

    Args:
        preset_scores: 预设的5维参数
        noise_level: 噪声水平（0-1）

    Returns:
        模拟的44题回答
    """
    validator = BFI44Validator()
    responses = {}

    # 维度到题目的映射
    dim_questions = {
        'E': [1,6,11,16,21,26,31,36],
        'A': [2,7,12,17,22,27,32,37,42],
        'C': [3,8,13,18,23,28,33,38,43],
        'N': [4,9,14,19,24,29,34,39],
        'O': [5,10,15,20,25,30,35,40,44]
    }

    reverse_questions = [2,6,8,9,12,18,21,23,24,27,31,34,35,37,41,43]

    for dim, qids in dim_questions.items():
        preset_score = preset_scores[dim]  # 0-1范围

        for qid in qids:
            # 将0-1的预设值映射到1-5的李克特量表
            # 0 → 1, 0.25 → 2, 0.5 → 3, 0.75 → 4, 1.0 → 5
            base_response = 1 + preset_score * 4

            # 如果是反向题，需要反转
            if qid in reverse_questions:
                base_response = 6 - base_response

            # 加入随机噪声
            noise = np.random.normal(0, noise_level * 2)
            response = base_response + noise

            # 限制在1-5范围
            response = max(1, min(5, response))
            responses[qid] = int(round(response))

    return responses


if __name__ == "__main__":
    # 测试代码
    print("BFI-44验证系统测试\n")

    # 加载Alice的预设参数
    preset_path = Path(__file__).parent.parent / "data" / "personas" / "preset_personality.json"
    with open(preset_path, 'r') as f:
        alice_preset = json.load(f)

    preset_scores = {
        'O': alice_preset['big_five_parameters']['O']['value'],
        'C': alice_preset['big_five_parameters']['C']['value'],
        'E': alice_preset['big_five_parameters']['E']['value'],
        'A': alice_preset['big_five_parameters']['A']['value'],
        'N': alice_preset['big_five_parameters']['N']['value']
    }

    print(f"Alice预设参数: {preset_scores}\n")

    # 模拟回答
    print("模拟Alice回答BFI-44...")
    simulated_responses = simulate_agent_responses(preset_scores, noise_level=0.1)

    # 创建验证器
    validator = BFI44Validator()

    # 生成报告
    report_path = validator.generate_report(
        agent_name="Alice Chen",
        test_type="pretest_simulation",
        responses=simulated_responses,
        preset_scores=preset_scores
    )

    # 打印摘要
    validator.print_validation_summary(report_path)
    print(f"\n报告已保存: {report_path}")
