"""
实验1：收敛性验证 (RQ1)
"""

from simulated_legacy.agents.interactions import run_experiment
from utils.data_processor import DataProcessor
from utils.config import Config
from pathlib import Path

def run(user_name='alice', num_days=100):
    """运行收敛性实验"""
    
    # 运行交互循环
    data = run_experiment('convergence', user_name, num_days)
    
    # 获取日志文件（最新的）
    log_dir = Path(f"experiments/results/convergence_{user_name}")
    log_file = sorted(log_dir.glob("*.log"))[-1]
    
    # 数据处理和可视化
    processor = DataProcessor(log_file)
    user_config = Config.get_user_config(user_name)
    
    # 生成图表
    processor.plot_convergence(user_config)
    
    # 获取指标
    metrics = processor.get_metrics(user_config)
    print(f"\n✓ 收敛性实验结果:")
    print(f"  - 最终E误差: {metrics['e_error_final']:.4f}")
    print(f"  - 最终O误差: {metrics['o_error_final']:.4f}")
    print(f"  - 收敛天数: {metrics['converge_day']}")
    
    return metrics
