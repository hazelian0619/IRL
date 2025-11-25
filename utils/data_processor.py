"""
数据处理和可视化
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class DataProcessor:
    def __init__(self, log_file):
        with open(log_file, 'r') as f:
            self.data = json.load(f)
    
    def plot_convergence(self, user_config, output_path='figures/'):
        """绘制收敛曲线"""
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        days = [d['day'] for d in self.data]
        e_est = [d['e_est'] for d in self.data]
        o_est = [d['o_est'] for d in self.data]
        
        e_true = user_config['E']
        o_true = user_config['O']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # E维度
        ax1.plot(days, e_est, 'b-', label='E_est', linewidth=2)
        ax1.axhline(y=e_true, color='r', linestyle='--', label=f'E_true={e_true}')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Extraversion (E)')
        ax1.set_ylim([0, 1])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Extraversion Learning Curve')
        
        # O维度
        ax2.plot(days, o_est, 'g-', label='O_est', linewidth=2)
        ax2.axhline(y=o_true, color='r', linestyle='--', label=f'O_true={o_true}')
        ax2.set_xlabel('Day')
        ax2.set_ylabel('Openness (O)')
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Openness Learning Curve')
        
        plt.tight_layout()
        plt.savefig(f'{output_path}convergence_{user_config["name"]}.png', dpi=150)
        print(f"✓ 图表已保存: {output_path}convergence_{user_config['name']}.png")
        plt.close()
    
    def get_metrics(self, user_config):
        """计算指标"""
        e_true = user_config['E']
        o_true = user_config['O']
        
        e_est_final = self.data[-1]['e_est']
        o_est_final = self.data[-1]['o_est']
        
        e_error = abs(e_est_final - e_true)
        o_error = abs(o_est_final - o_true)
        
        # 找到收敛时刻（误差<0.1）
        converge_day = None
        for i, d in enumerate(self.data):
            e_error_i = abs(d['e_est'] - e_true)
            o_error_i = abs(d['o_est'] - o_true)
            if e_error_i < 0.1 and o_error_i < 0.1:
                converge_day = d['day']
                break
        
        return {
            'e_error_final': float(e_error),
            'o_error_final': float(o_error),
            'converge_day': converge_day,
            'total_days': len(self.data),
        }
