"""
多模态反馈融合
"""

import numpy as np
from utils.config import Config

class FeedbackFusion:
    def __init__(self):
        self.weights = Config.FEEDBACK_WEIGHTS
    
    def fuse(self, explicit, behavior=None, emotion=None, social=None):
        """
        融合多模态反馈
        
        Args:
            explicit: 显式反馈 (1-5星)
            behavior: 行为反馈 (0-1)
            emotion: 情感反馈 (0-1)
            social: 社交反馈 (0-1)
        
        Returns:
            融合后的反馈 (float)
        """
        
        # 归一化显式反馈到0-2范围
        explicit_normalized = (explicit - 1) / 4 * 2
        
        # 默认值
        behavior = behavior or 0.5
        emotion = emotion or 0.5
        social = social or 0.5
        
        # 加权融合
        fused = (
            self.weights['explicit'] * explicit_normalized +
            self.weights['behavior'] * behavior +
            self.weights['emotion'] * emotion +
            self.weights['social'] * social
        )
        
        return float(fused)

feedback_fusion = FeedbackFusion()

def fuse_feedback(*args, **kwargs):
    return feedback_fusion.fuse(*args, **kwargs)
