"""
羊毛党攻击模拟器
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置编码（仅当stdout未关闭时）
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass  # stdout可能已关闭或不是缓冲区

import requests
import time
import random
import threading
from datetime import datetime
from utils.data_generator import DataGenerator

class Attacker:
    """请求模拟器 - 包含正常用户和羊毛党"""
    
    def __init__(self, api_url='http://localhost:5000/api/register', thread_count=10, requests_per_thread=20, attack_ratio=0.3):
        """
        初始化模拟器
        
        Args:
            api_url: API地址
            thread_count: 线程数
            requests_per_thread: 每线程请求数
            attack_ratio: 攻击请求比例（0-1，默认0.3即30%为攻击请求）
        """
        self.api_url = api_url
        self.thread_count = int(thread_count)
        self.requests_per_thread = int(requests_per_thread)
        self.attack_ratio = float(attack_ratio)  # 攻击请求比例
        
        # 验证参数
        if self.thread_count < 1:
            raise ValueError(f"线程数必须 >= 1，当前: {thread_count}")
        if self.requests_per_thread < 1:
            raise ValueError(f"每线程请求数必须 >= 1，当前: {requests_per_thread}")
        if not 0 <= self.attack_ratio <= 1:
            raise ValueError(f"攻击比例必须在0-1之间，当前: {attack_ratio}")
        self.generator = DataGenerator()
        self.results = []
        self.lock = threading.Lock()
        self.running = False
        self.stats = {
            'total_requests': 0,
            'normal_requests': 0,  # 正常请求数
            'attack_requests': 0,  # 攻击请求数
            'success_count': 0,    # 通过数
            'blocked_count': 0,    # 拦截数
            'review_count': 0,     # 审核数
        }
    
    def simulate_request(self, thread_id):
        """模拟请求（包含正常用户和羊毛党）"""
        for i in range(self.requests_per_thread):
            if not self.running:
                break
            
            # 根据attack_ratio决定生成正常请求还是攻击请求
            is_attack = random.random() < self.attack_ratio
            
            if is_attack:
                # 生成攻击请求数据（模拟羊毛党行为）
                request_data = {
                    'phone': self.generator.generate_phone(is_blacklist=random.random() < 0.2),
                    'ip': self.generator.generate_ip(reuse_probability=0.4),  # 高复用率
                    'device_id': self.generator.generate_device_id(reuse_probability=0.4),
                    'user_agent': self.generator.generate_user_agent(),
                    'device_fingerprint': self.generator.generate_device_fingerprint(),
                    'timestamp': time.time(),
                    'behavior': self.generator.generate_behavior_sequence(is_normal=False),  # 异常行为
                    '_true_label': 'attack',  # 真实标签（上帝视角）
                }
                request_type = 'attack'
            else:
                # 生成正常请求数据
                request_data = {
                    'phone': self.generator.generate_phone(is_blacklist=False),
                    'ip': self.generator.generate_ip(reuse_probability=0.1),  # 低复用率
                    'device_id': self.generator.generate_device_id(reuse_probability=0.1),
                    'user_agent': self.generator.generate_user_agent(),
                    'device_fingerprint': self.generator.generate_device_fingerprint(),
                    'timestamp': time.time(),
                    'behavior': self.generator.generate_behavior_sequence(is_normal=True),  # 正常行为
                    '_true_label': 'normal',  # 真实标签（上帝视角）
                }
                request_type = 'normal'
            
            try:
                # 发送请求
                response = requests.post(
                    self.api_url,
                    json=request_data,
                    timeout=5
                )
                
                result = response.json()
                
                with self.lock:
                    self.stats['total_requests'] += 1
                    
                    # 统计请求类型
                    if request_type == 'attack':
                        self.stats['attack_requests'] += 1
                    else:
                        self.stats['normal_requests'] += 1
                    
                    # 统计风险等级
                    risk_level = result.get('risk_level', 'unknown')
                    if risk_level == 'pass':
                        self.stats['success_count'] += 1
                    elif risk_level == 'review':
                        self.stats['review_count'] += 1
                    else:
                        self.stats['blocked_count'] += 1
                    
                    self.results.append({
                        'thread_id': thread_id,
                        'request_id': i,
                        'request_type': request_type,
                        'timestamp': datetime.now().isoformat(),
                        'request_data': request_data,
                        'response': result,
                    })
                
                # 模拟请求间隔（1-3秒，加快速度）
                interval = random.uniform(1, 3)
                # 检查是否需要停止
                for _ in range(int(interval * 10)):
                    if not self.running:
                        break
                    time.sleep(0.1)
                
            except Exception as e:
                print(f"线程 {thread_id} 请求 {i} 失败: {e}")
                time.sleep(1)
    
    def start_attack(self):
        """开始模拟（包含正常用户和羊毛党）"""
        total_requests = self.thread_count * self.requests_per_thread
        expected_attacks = int(total_requests * self.attack_ratio)
        expected_normal = total_requests - expected_attacks
        
        print(f"\n{'='*60}")
        print(f"开始请求模拟（正常用户 + 羊毛党）")
        print(f"线程数: {self.thread_count}")
        print(f"每线程请求数: {self.requests_per_thread}")
        print(f"总请求数: {total_requests}")
        print(f"预期正常请求: {expected_normal} ({100*(1-self.attack_ratio):.1f}%)")
        print(f"预期攻击请求: {expected_attacks} ({100*self.attack_ratio:.1f}%)")
        print(f"{'='*60}\n")
        
        self.running = True
        self.results = []
        self.stats = {
            'total_requests': 0,
            'success_count': 0,
            'blocked_count': 0,
            'review_count': 0,
        }
        
        threads = []
        start_time = time.time()
        
        # 启动多个线程
        for i in range(self.thread_count):
            thread = threading.Thread(target=self.simulate_request, args=(i,))
            thread.start()
            threads.append(thread)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        duration = end_time - start_time
        
        self.running = False
        
        # 打印统计
        total = self.stats['total_requests']
        print(f"\n{'='*60}")
        print(f"模拟完成！")
        print(f"耗时: {duration:.2f} 秒")
        print(f"\n【请求统计】")
        print(f"总请求数: {total}")
        print(f"  正常请求: {self.stats['normal_requests']} ({self.stats['normal_requests']/max(total,1)*100:.1f}%)")
        print(f"  攻击请求: {self.stats['attack_requests']} ({self.stats['attack_requests']/max(total,1)*100:.1f}%)")
        print(f"\n【风险等级统计】")
        print(f"通过 (pass): {self.stats['success_count']} ({self.stats['success_count']/max(total,1)*100:.1f}%)")
        print(f"审核 (review): {self.stats['review_count']} ({self.stats['review_count']/max(total,1)*100:.1f}%)")
        print(f"拦截 (reject): {self.stats['blocked_count']} ({self.stats['blocked_count']/max(total,1)*100:.1f}%)")
        print(f"{'='*60}\n")
        
        return {
            'duration': duration,
            'stats': self.stats,
            'results': self.results,
        }
    
    def stop_attack(self):
        """停止攻击"""
        self.running = False

