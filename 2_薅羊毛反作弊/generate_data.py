"""
数据生成脚本 - 生成并保存完整的原始训练数据
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import random
import time
import pandas as pd
from tqdm import tqdm
from utils.data_generator import DataGenerator
from config import DATA_DIR

def generate_and_save_data(n_samples=12000, normal_ratio=0.8, output_file=None):
    """生成完整的原始训练数据并保存到CSV"""
    if output_file is None:
        output_file = os.path.join(DATA_DIR, 'training_data.csv')
    
    print("=" * 60)
    print("生成训练数据（完整原始数据）")
    print("=" * 60)
    
    generator = DataGenerator()
    data_list = []
    n_normal = int(n_samples * normal_ratio)
    n_abnormal = n_samples - n_normal
    
    # 生成正常样本
    print(f"\n生成 {n_normal} 个正常样本...")
    for i in tqdm(range(n_normal), desc="正常样本", unit="个"):
        timestamp = time.time() + random.randint(0, 86400)
        request_data = {
            'phone': generator.generate_phone(is_blacklist=False),
            'ip': generator.generate_ip(reuse_probability=0.1),
            'device_id': generator.generate_device_id(reuse_probability=0.1),
            'user_agent': generator.generate_user_agent(),
            'device_fingerprint': generator.generate_device_fingerprint(),
            'timestamp': timestamp,
            'behavior': generator.generate_behavior_sequence(is_normal=True),
            'label': 0,  # 正常用户
        }
        data_list.append(request_data)
    
    # 生成异常样本
    print(f"\n生成 {n_abnormal} 个异常样本...")
    for i in tqdm(range(n_abnormal), desc="异常样本", unit="个"):
        timestamp = time.time() + random.randint(0, 86400)
        request_data = {
            'phone': generator.generate_phone(is_blacklist=random.random() < 0.3),
            'ip': generator.generate_ip(reuse_probability=0.5),
            'device_id': generator.generate_device_id(reuse_probability=0.5),
            'user_agent': generator.generate_user_agent(),
            'device_fingerprint': generator.generate_device_fingerprint(),
            'timestamp': timestamp,
            'behavior': generator.generate_behavior_sequence(is_normal=False),
            'label': 1,  # 羊毛党
        }
        data_list.append(request_data)
    
    # 转换为DataFrame，处理嵌套字典
    records = []
    for item in data_list:
        record = {
            'label': item['label'],
            'phone': item['phone'],
            'ip': item['ip'],
            'device_id': item['device_id'],
            'user_agent': item['user_agent'],
            'timestamp': item['timestamp'],
            # 设备指纹
            'canvas_fingerprint': item['device_fingerprint']['canvas_fingerprint'],
            'fonts': item['device_fingerprint']['fonts'],
            'screen_resolution': item['device_fingerprint']['screen_resolution'],
            'timezone': item['device_fingerprint']['timezone'],
            'language': item['device_fingerprint']['language'],
            # 行为特征
            'page_stay_time': item['behavior']['page_stay_time'],
            'click_count': item['behavior']['click_count'],
            'scroll_count': item['behavior']['scroll_count'],
            'path_entropy': item['behavior']['path_entropy'],
            'mouse_trajectory_entropy': item['behavior']['mouse_trajectory_entropy'],
            'request_frequency': item['behavior']['request_frequency'],
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    df.to_csv(output_file, index=False)
    
    print(f"\n[OK] 数据已保存: {output_file}")
    print(f"  总样本数: {len(df)}")
    print(f"  正常样本: {len(df[df['label']==0])}")
    print(f"  异常样本: {len(df[df['label']==1])}")
    print(f"  字段数: {len(df.columns)}")
    # 字段名到中文含义的映射
    field_names = {
        'label': '标签',
        'phone': '手机号',
        'ip': 'IP地址',
        'device_id': '设备ID',
        'user_agent': '用户代理',
        'timestamp': '时间戳',
        'canvas_fingerprint': 'Canvas指纹',
        'fonts': '字体',
        'screen_resolution': '屏幕分辨率',
        'timezone': '时区',
        'language': '语言',
        'page_stay_time': '页面停留时间',
        'click_count': '点击次数',
        'scroll_count': '滚动次数',
        'path_entropy': '路径熵',
        'mouse_trajectory_entropy': '鼠标轨迹熵',
        'request_frequency': '请求频率'
    }
    
    # 格式化字段列表，显示字段名和中文含义
    field_list = [f"{col}({field_names.get(col, col)})" for col in df.columns.tolist()]
    print(f"\n字段列表: {', '.join(field_list)}")
    
    return output_file

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='生成训练数据')
    parser.add_argument('--samples', type=int, default=20000, help='总样本数')
    parser.add_argument('--ratio', type=float, default=0.95, help='正常样本比例')
    parser.add_argument('--output', type=str, default=None, help='输出文件路径')
    args = parser.parse_args()
    
    generate_and_save_data(n_samples=args.samples, normal_ratio=args.ratio, output_file=args.output)
