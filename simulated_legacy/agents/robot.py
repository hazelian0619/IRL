"""
陪伴机器人Robot
"""

import numpy as np
from ..learning.online_irl import OnlineIRL
from ..learning.feedback_fusion import fuse_feedback
from ..learning.explanation import generate_explanation
from utils.config import Config

class Robot:
    def __init__(self):
        self.irl = OnlineIRL()
        self.decision_history = []
        self.num_interactions = 0
    
    def decide(self, day):
        """
        决策：选择陪伴方式
        
        Returns:
            action: {'social': 0/1, 'novelty': 0/1}
        """
        
        # 计算4个动作的吸引力
        actions = [
            {'name': 'A1', 'social': 1, 'novelty': 1},
            {'name': 'A2', 'social': 1, 'novelty': 0},
            {'name': 'A3', 'social': 0, 'novelty': 1},
            {'name': 'A4', 'social': 0, 'novelty': 0},
        ]
        
        scores = []
        for a in actions:
            score = self.irl.predict_feedback(a)
            scores.append(score)
        
        # 选择最高分的动作
        best_idx = np.argmax(scores)
        chosen_action = actions[best_idx]
        
        self.decision_history.append({
            'day': day,
            'action': chosen_action,
            'scores': scores,
        })
        
        return chosen_action
    
    def learn(self, action, explicit_feedback, implicit_feedback, day):
        """
        从反馈学习
        
        Args:
            action: Robot的决策
            explicit_feedback: Alice的评分(1-5星)
            implicit_feedback: {'behavior': 0-1, 'emotion': 0-1, 'social': 0-1}
            day: 当前天数
        """
        
        # 融合反馈
        fused = fuse_feedback(
            explicit=explicit_feedback,
            behavior=implicit_feedback.get('behavior', 0.5),
            emotion=implicit_feedback.get('emotion', 0.5),
            social=implicit_feedback.get('social', 0.5),
        )
        
        # IRL更新
        self.num_interactions += 1
        e_est, o_est, error = self.irl.update(
            action,
            fused,
            day,
            num_samples=self.num_interactions
        )
        
        return e_est, o_est, fused, error
    
    def generate_explanation(self, action=None):
        """生成解释"""
        state = self.irl.get_state()
        return generate_explanation(state['E'], state['O'], action)
    
    def get_state(self):
        """获取当前理解"""
        return self.irl.get_state()
