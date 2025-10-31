"""
配置文件
"""
import os

# 基础配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
# 训练产物（模型文件）改为保存在代码目录下的 models/models 目录
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'models', 'models')
LOGS_DIR = os.path.join(DATA_DIR, 'log')
TRAINING_DATA_FILE = os.path.join(DATA_DIR, 'training_data.csv')

# 确保目录存在
for dir_path in [DATA_DIR, ARTIFACTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# API配置
API_HOST = '0.0.0.0'
API_PORT = 5000
API_DEBUG = True

# 风控配置
RISK_THRESHOLD_LOW = 0.5   # 低风险阈值
RISK_THRESHOLD_HIGH = 0.8  # 高风险阈值

# 模型配置（保存/加载路径）
MODEL_CONFIG = {
    'xgb_model_path': os.path.join(ARTIFACTS_DIR, 'xgb_model.pkl'),
    'lstm_model_path': os.path.join(ARTIFACTS_DIR, 'lstm_model.pkl'),
    'isolation_forest_path': os.path.join(ARTIFACTS_DIR, 'isolation_forest.pkl'),
    'ensemble_weights': {
        'supervised': 0.6,
        'unsupervised': 0.4
    }
}

# 攻击模拟配置
ATTACK_CONFIG = {
    'thread_count': 10,        # 并发线程数
    'requests_per_thread': 20, # 每个线程的请求数
    'request_interval': (5, 10), # 请求间隔（秒）
    'use_proxy': True,         # 是否使用代理IP
    'proxy_pool_size': 50,     # 代理池大小
}

# 特征配置
FEATURE_CONFIG = {
    'time_window': 3600,       # 时间窗口（秒）
    'max_registrations_per_ip': 5,  # IP注册上限
    'max_registrations_per_device': 3, # 设备注册上限
}

