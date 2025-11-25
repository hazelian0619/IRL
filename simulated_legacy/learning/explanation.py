"""
解释生成模块
"""

class ExplanationGenerator:
    def __init__(self):
        self.templates = {
            'social': "你很喜欢社交活动，这就是为什么我建议这个。",
            'novelty': "你对新奇体验持开放态度，所以我选择了这个。",
            'combined': "基于我对你的理解，我认为你会喜欢这个。",
        }
    
    def generate(self, e_est, o_est, action=None):
        """
        生成解释
        
        Args:
            e_est: 外向性估计
            o_est: 开放性估计
            action: 当前动作 (可选)
        
        Returns:
            解释文本 (str)
        """
        
        if action:
            explanation = f"我建议这个是因为："
            
            if action.get('social'):
                explanation += f" 你的外向性特征很强({e_est:.2f})，喜欢社交活动。"
            
            if action.get('novelty'):
                explanation += f" 你对新奇体验很开放({o_est:.2f})。"
            
            return explanation
        
        return self.templates['combined']

explanation_generator = ExplanationGenerator()

def generate_explanation(*args, **kwargs):
    return explanation_generator.generate(*args, **kwargs)
