"""
在线IRL学习模块
"""

import numpy as np
from utils.config import Config

class OnlineIRL:
    def __init__(self):
        self.E_est = 0.5
        self.O_est = 0.5
        self.alpha = Config.IRL['learning_rate']
        self.tau = Config.IRL['tau']
        self.kappa = Config.IRL['kappa']
        self.history = []
    
    def predict_feedback(self, action):
        """预测反馈"""
        return self.E_est * action['social'] + self.O_est * action['novelty']
    
    def compute_decay_weight(self, day):
        """计算时序衰减权重"""
        return np.exp(-day / self.tau)
    
    def compute_confidence_weight(self, num_samples):
        """计算置信度权重"""
        return 1 - np.exp(-num_samples / self.kappa)
    
    def update(self, action, feedback, day, num_samples=None):
        """
        在线IRL更新
        
        Args:
            action: {'social': 0/1, 'novelty': 0/1}
            feedback: 融合反馈值 (float)
            day: 当前天数
            num_samples: 总反馈样本数
        """
        
        # 预测反馈
        predicted = self.predict_feedback(action)
        
        # 计算误差
        error = feedback - predicted
        
        # 时序衰减权重
        decay_w = self.compute_decay_weight(day)
        
        # 置信度权重
        conf_w = self.compute_confidence_weight(num_samples) if num_samples else 1.0
        
        # 综合权重
        weight = decay_w * conf_w
        
        # 更新估计
        self.E_est += self.alpha * weight * error * action['social']
        self.O_est += self.alpha * weight * error * action['novelty']
        
        # 限制范围
        self.E_est = np.clip(self.E_est, 0, 1)
        self.O_est = np.clip(self.O_est, 0, 1)
        
        # 记录历史
        self.history.append({
            'day': day,
            'e_est': self.E_est,
            'o_est': self.O_est,
            'error': error,
            'decay_w': decay_w,
        })
        
        return self.E_est, self.O_est, error
    
    def get_state(self):
        """获取当前状态"""
        return {'E': self.E_est, 'O': self.O_est}
