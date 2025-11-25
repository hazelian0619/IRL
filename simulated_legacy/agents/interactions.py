"""
Alice和Robot的交互接口
"""

from .alice import create_user
from .robot import Robot
from utils import logger

def run_daily_interaction(robot, alice, day):
    """
    每一天的交互循环
    
    Returns:
        (e_est, o_est, feedback, error)
    """
    
    # 早上：Alice在Town中活动（这里用隐式反馈代理）
    implicit_feedback = alice.get_implicit_feedback()
    
    # 晚上：Robot决策
    action = robot.decide(day)
    
    # Alice给出显式反馈
    explicit_feedback = alice.evaluate_suggestion(action)
    
    # Robot学习
    e_est, o_est, fused, error = robot.learn(
        action,
        explicit_feedback,
        implicit_feedback,
        day
    )
    
    # 记录日志
    logger.log_day(day, e_est, o_est, fused, error)
    
    return e_est, o_est, fused, error

def run_experiment(exp_type, user_name, num_days):
    """
    运行实验
    
    Args:
        exp_type: 'convergence', 'decay', 'understanding', 'personalization'
        user_name: 'alice', 'bob', 'carol'
        num_days: 模拟天数
    """
    
    print(f"\n{'='*60}")
    print(f"实验: {exp_type} | 用户: {user_name} | 天数: {num_days}")
    print(f"{'='*60}\n")
    
    # 初始化
    alice = create_user(user_name)
    robot = Robot()
    
    # 主循环
    for day in range(1, num_days + 1):
        run_daily_interaction(robot, alice, day)
        
        # Day 50: 突变实验（仅在decay实验中）
        if day == 50 and exp_type == 'decay':
            alice.change_personality(0.5, 0.3)
    
    # 保存结果
    logger.save()
    
    return logger.get_data()
