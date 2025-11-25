"""
BFI-44 Interviewer - 向Agent提问并收集真实回答
通过LLM生成Agent的自然语言回答，提取量化分数
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
import sys

# 添加Backend路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'external_town' / 'reverie' / 'backend_server'))

try:
    from persona.prompt_template.gpt_structure import ChatGPT_request
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️  Warning: LLM not available, will use fallback method")


class BFIInterviewer:
    """BFI-44问卷施测系统"""

    def __init__(self, questionnaire_path=None):
        """初始化"""
        if questionnaire_path is None:
            questionnaire_path = Path(__file__).parent.parent / "data" / "bfi44_questionnaire.json"

        with open(questionnaire_path, 'r', encoding='utf-8') as f:
            self.questionnaire = json.load(f)

        self.questions = self.questionnaire['questions']

    def load_agent_profile(self, agent_name: str) -> dict:
        """加载Agent的人格档案"""
        # 从Alice的传记和预设参数中提取
        personas_dir = Path(__file__).parent.parent / "data" / "personas"

        # 加载传记
        bio_path = personas_dir / "alice_biography_prompt.txt"
        with open(bio_path, 'r', encoding='utf-8') as f:
            biography = f.read()

        # 加载预设参数
        preset_path = personas_dir / "preset_personality.json"
        with open(preset_path, 'r', encoding='utf-8') as f:
            preset = json.load(f)

        return {
            'name': agent_name,
            'biography': biography,
            'preset_scores': {
                'O': preset['big_five_parameters']['O']['value'],
                'C': preset['big_five_parameters']['C']['value'],
                'E': preset['big_five_parameters']['E']['value'],
                'A': preset['big_five_parameters']['A']['value'],
                'N': preset['big_five_parameters']['N']['value']
            }
        }

    def construct_prompt(self, agent_profile: dict, question: dict) -> str:
        """构建提问Prompt"""
        prompt = f"""You are {agent_profile['name']}. Here is your background:

{agent_profile['biography'][:500]}...

Now, please answer this personality assessment question honestly based on your character:

Question: "{question['en']}"
(中文: {question['zh']})

Please respond with:
1. Your natural language answer explaining your perspective (2-3 sentences)
2. A rating from 1-5:
   1 = Strongly Disagree (非常不同意)
   2 = Disagree a Little (有点不同意)
   3 = Neither Agree nor Disagree (既不同意也不反对)
   4 = Agree a Little (有点同意)
   5 = Strongly Agree (非常同意)

Format your response as:
Rating: [1-5]
Explanation: [Your explanation]
"""
        return prompt

    def extract_score_from_response(self, response: str) -> tuple:
        """从LLM回答中提取分数"""
        # 提取Rating
        rating_match = re.search(r'Rating:\s*(\d)', response, re.IGNORECASE)
        if rating_match:
            score = int(rating_match.group(1))
        else:
            # 尝试在文本中找数字
            numbers = re.findall(r'\b([1-5])\b', response)
            if numbers:
                score = int(numbers[0])
            else:
                score = 3  # 默认中立

        # 提取解释
        explanation_match = re.search(r'Explanation:\s*(.+)', response, re.DOTALL | re.IGNORECASE)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        else:
            explanation = response.strip()

        return score, explanation

    def ask_question(self, agent_profile: dict, question: dict, use_llm: bool = True) -> dict:
        """向Agent提问单个问题"""

        if use_llm and LLM_AVAILABLE:
            # 使用真实LLM
            prompt = self.construct_prompt(agent_profile, question)
            try:
                response = ChatGPT_request(prompt)
                score, explanation = self.extract_score_from_response(response)
                method = "LLM"
            except Exception as e:
                print(f"⚠️  LLM failed for Q{question['id']}: {e}")
                # Fallback to rule-based
                score, explanation = self.fallback_answer(agent_profile, question)
                method = "Fallback"
        else:
            # 基于规则的回答（用于测试或LLM不可用时）
            score, explanation = self.fallback_answer(agent_profile, question)
            method = "Fallback"

        return {
            'question_id': question['id'],
            'question_en': question['en'],
            'question_zh': question['zh'],
            'dimension': question['dimension'],
            'reverse_scored': question['reverse_scored'],
            'score': score,
            'explanation': explanation,
            'method': method
        }

    def fallback_answer(self, agent_profile: dict, question: dict) -> tuple:
        """基于预设参数的回答（Fallback）"""
        preset_scores = agent_profile['preset_scores']
        dimension = question['dimension']
        is_reverse = question['reverse_scored']

        # 获取该维度的预设分数（0-1）
        preset_value = preset_scores[dimension]

        # 转换为1-5的李克特量表
        # 0 -> 1, 0.25 -> 2, 0.5 -> 3, 0.75 -> 4, 1.0 -> 5
        base_score = 1 + preset_value * 4

        # 如果是反向题，翻转
        if is_reverse:
            base_score = 6 - base_score

        # 加入小噪声
        import random
        noise = random.gauss(0, 0.3)
        final_score = max(1, min(5, base_score + noise))

        score = int(round(final_score))

        # 生成简单解释
        if score >= 4:
            explanation = f"I agree with this statement. It aligns with my personality ({dimension}={preset_value:.2f})."
        elif score <= 2:
            explanation = f"I don't really agree with this. It doesn't match how I see myself ({dimension}={preset_value:.2f})."
        else:
            explanation = f"I'm neutral about this statement."

        return score, explanation

    def complete_questionnaire(self, agent_name: str, use_llm: bool = True) -> dict:
        """完成完整的44题问卷"""

        print(f"{'='*60}")
        print(f"BFI-44 Interview: {agent_name}")
        print(f"{'='*60}")
        print(f"Method: {'Real LLM' if (use_llm and LLM_AVAILABLE) else 'Fallback (Rule-based)'}")
        print()

        # 加载Agent档案
        agent_profile = self.load_agent_profile(agent_name)

        # 收集所有回答
        responses = []

        for i, question in enumerate(self.questions, 1):
            print(f"[{i}/44] Q{question['id']}: {question['dimension']} - ", end='', flush=True)

            answer = self.ask_question(agent_profile, question, use_llm)
            responses.append(answer)

            print(f"Score: {answer['score']} ({answer['method']})")

            # 每10题显示进度
            if i % 10 == 0:
                print(f"  Progress: {i}/44 completed\n")

        print(f"\n✓ All 44 questions answered\n")

        # 生成完整报告
        report = {
            'agent_name': agent_name,
            'test_type': 'pretest_REAL',
            'test_date': datetime.now().isoformat(),
            'method': 'LLM' if (use_llm and LLM_AVAILABLE) else 'Fallback',
            'total_questions': 44,
            'responses_detailed': responses,
            'responses_scores': {str(r['question_id']): r['score'] for r in responses},
            'agent_profile': {
                'biography_length': len(agent_profile['biography']),
                'preset_scores': agent_profile['preset_scores']
            }
        }

        return report

    def save_report(self, report: dict, output_dir: str = None) -> str:
        """保存测试报告"""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "validation"
            output_dir.mkdir(exist_ok=True)

        method_tag = report['method']
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{report['agent_name'].replace(' ', '_')}_{report['test_type']}_{method_tag}_{timestamp}.json"

        output_path = Path(output_dir) / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"✓ Report saved: {output_path}")
        return str(output_path)


def main():
    """主函数 - 执行BFI-44前测"""

    interviewer = BFIInterviewer()

    # 询问使用方式
    print("\nBFI-44 Pretest Configuration")
    print("="*60)
    print("1. Use Real LLM (Llama) - Alice generates natural responses")
    print("2. Use Fallback (Rule-based) - Faster, for testing")
    print()

    if LLM_AVAILABLE:
        choice = input("Select method (1/2) [default: 1]: ").strip() or "1"
        use_llm = (choice == "1")
    else:
        print("⚠️  LLM not available, using Fallback method")
        use_llm = False

    print()

    # 执行问卷
    report = interviewer.complete_questionnaire("Alice Chen", use_llm=use_llm)

    # 保存报告
    report_path = interviewer.save_report(report)

    # 显示摘要
    print("\n" + "="*60)
    print("BFI-44 Pretest Complete")
    print("="*60)
    print(f"Agent: {report['agent_name']}")
    print(f"Method: {report['method']}")
    print(f"Questions answered: {report['total_questions']}")
    print(f"Report: {report_path}")
    print()
    print("Next step: Run validation analysis")
    print("  python agents/bfi_validator.py")


if __name__ == "__main__":
    main()
