"""
全局配置文件
"""

import yaml
from pathlib import Path

class Config:
    # 虚拟用户配置
    USERS = {
        'alice': {'E': 0.8, 'O': 0.6, 'name': 'Alice'},
        'bob': {'E': 0.2, 'O': 0.3, 'name': 'Bob'},
        'carol': {'E': 0.5, 'O': 0.5, 'name': 'Carol'},
    }
    
    # IRL参数
    IRL = {
        'learning_rate': 0.1,
        'tau': 5.0,  # 时序衰减参数
        'kappa': 5.0,  # 置信度参数
    }
    
    # 反馈融合权重
    FEEDBACK_WEIGHTS = {
        'explicit': 0.5,
        'behavior': 0.25,
        'emotion': 0.15,
        'social': 0.1,
    }
    
    # 时间加速配置
    TIME_SPEED = {
        'normal': 1,      # 无加速
        '30m': 2,         # 1秒 = 30分钟
        '1h': 60,         # 1秒 = 1小时
        '2h': 120,        # 1秒 = 2小时
    }
    
    # 默认值
    DEFAULT_SPEED = '1h'
    DEFAULT_DAYS = 100
    
    @classmethod
    def get_user_config(cls, user_name):
        return cls.USERS.get(user_name, cls.USERS['alice'])
    
    @classmethod
    def set_speed(cls, speed_name):
        if speed_name in cls.TIME_SPEED:
            cls.DEFAULT_SPEED = speed_name
            print(f"✓ 时间加速设置为: {speed_name}")
        else:
            print(f"✗ 未知的加速配置: {speed_name}")


# 提供模块级别的便捷函数，方便以`import config as cfg`的方式调用
def set_speed(speed_name):
    """设置全局时间加速配置"""
    return Config.set_speed(speed_name)


def get_speed_multiplier():
    """获取当前加速倍率"""
    return Config.TIME_SPEED.get(Config.DEFAULT_SPEED, 1)


def get_user_config(user_name):
    """获取指定虚拟用户的配置"""
    return Config.get_user_config(user_name)
