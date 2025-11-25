"""
虚拟用户Alice
"""

import random
from utils.config import Config

class Alice:
    def __init__(self, name, e_true, o_true):
        self.name = name
        self.E = e_true
        self.O = o_true
        self.memory = []
        self.satisfaction_history = []
    
    def evaluate_suggestion(self, action):
        """
        对Robot建议的评价
        
        Args:
            action: {'social': 0/1, 'novelty': 0/1}
        
        Returns:
            评分 (1-5星)
        """
        
        # 真实反馈函数
        raw_score = self.E * action['social'] + self.O * action['novelty']
        
        # 转换为星级 (1-5)
        star_rating = 1 + (raw_score / 2.0) * 4
        
        # 加入小噪声（模拟真实反馈的波动）
        noise = random.gauss(0, 0.1)
        star_rating = max(1, min(5, star_rating + noise))
        
        self.satisfaction_history.append(star_rating)
        self.memory.append({
            'action': action,
            'rating': star_rating,
        })
        
        return float(star_rating)
    
    def get_implicit_feedback(self):
        """
        模拟隐式反馈（从Town日志中提取的代理）
        
        Returns:
            {'behavior': 0-1, 'emotion': 0-1, 'social': 0-1}
        """
        
        if not self.satisfaction_history:
            return {'behavior': 0.5, 'emotion': 0.5, 'social': 0.5}
        
        recent_avg_satisfaction = sum(self.satisfaction_history[-5:]) / min(5, len(self.satisfaction_history))
        
        return {
            'behavior': min(1, recent_avg_satisfaction / 5.0),
            'emotion': min(1, recent_avg_satisfaction / 5.0),
            'social': min(1, (3 + self.E) / 5.0),  # 与外向性相关
        }
    
    def change_personality(self, new_e, new_o):
        """改变个性（用于Day 50的突变实验）"""
        self.E = new_e
        self.O = new_o
        print(f"✓ {self.name}的个性改变: E={new_e}, O={new_o}")

def create_user(user_name):
    """工厂函数"""
    config = Config.get_user_config(user_name)
    return Alice(config['name'], config['E'], config['O'])
