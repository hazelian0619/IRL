"""
日志记录工具
"""

import json
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.log_dir = Path(f"experiments/results/{exp_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.data = []
    
    def log_day(self, day, e_est, o_est, feedback, error):
        """记录每一天的数据"""
        record = {
            'day': day,
            'e_est': float(e_est),
            'o_est': float(o_est),
            'feedback': float(feedback),
            'error': float(error),
            'timestamp': datetime.now().isoformat(),
        }
        self.data.append(record)
        print(f"Day {day}: E={e_est:.3f}, O={o_est:.3f}, Feedback={feedback:.2f}")
    
    def save(self):
        """保存日志到JSON"""
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"✓ 日志已保存: {self.log_file}")
    
    def get_data(self):
        return self.data

logger_instance = None

def setup(exp_name):
    global logger_instance
    logger_instance = Logger(exp_name)
    return logger_instance

def log_day(*args, **kwargs):
    if logger_instance:
        logger_instance.log_day(*args, **kwargs)

def save():
    if logger_instance:
        logger_instance.save()

def get_data():
    if logger_instance:
        return logger_instance.get_data()
    return []
