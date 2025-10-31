"""
数据生成器 - 增强版（生成更难区分的数据）
"""
import random
import time
from faker import Faker
import hashlib

fake = Faker('zh_CN')

class DataGenerator:
    """数据生成器 - 增强版"""
    
    def __init__(self):
        self.fake = fake
        self.device_pool = []
        self.ip_pool = []
        self._init_pools()
    
    def _init_pools(self):
        """初始化设备池和IP池"""
        for _ in range(200):
            device_id = hashlib.md5(f"{random.random()}{time.time()}".encode()).hexdigest()
            self.device_pool.append(device_id)
        
        for _ in range(100):
            ip = fake.ipv4()
            self.ip_pool.append(ip)
    
    def generate_phone(self, is_blacklist=False):
        """生成手机号"""
        if is_blacklist:
            # 增加更多类型的黑号
            prefix_choices = [
                ('199', 0.4),  # 虚拟号段
                ('17', 0.3),   # 虚拟号段
                ('15', 0.2),   # 普通号段（模拟被盗用）
                ('13', 0.1),   # 普通号段
            ]
            prefix = random.choices([p[0] for p in prefix_choices], 
                                  weights=[p[1] for p in prefix_choices])[0]
            return f"{prefix}{random.randint(100000000, 999999999)}"[:11]
        return fake.phone_number()
    
    def generate_device_id(self, reuse_probability=0.3):
        """生成设备ID"""
        if random.random() < reuse_probability and self.device_pool:
            return random.choice(self.device_pool)
        device_id = hashlib.md5(f"{random.random()}{time.time()}".encode()).hexdigest()
        self.device_pool.append(device_id)
        return device_id
    
    def generate_ip(self, reuse_probability=0.3):
        """生成IP地址"""
        if random.random() < reuse_probability and self.ip_pool:
            return random.choice(self.ip_pool)
        ip = fake.ipv4()
        self.ip_pool.append(ip)
        return ip
    
    def generate_user_agent(self):
        """生成User-Agent"""
        agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148',
            'Mozilla/5.0 (Linux; Android 12; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
        return random.choice(agents)
    
    def generate_device_fingerprint(self):
        """生成设备指纹"""
        fonts_list = [
            'Arial, Helvetica, sans-serif',
            'Times New Roman, serif',
            'Courier New, monospace',
            'Verdana, Geneva, sans-serif',
            'Georgia, serif',
            'Trebuchet MS, sans-serif',
        ]
        
        canvas_hash = hashlib.md5(f"{random.random()}{time.time()}".encode()).hexdigest()[:16]
        
        return {
            'canvas_fingerprint': canvas_hash,
            'fonts': random.choice(fonts_list),
            'screen_resolution': f"{random.choice([1920, 1366, 1440, 1536, 1600])}x{random.choice([1080, 768, 900, 864, 1024])}",
            'timezone': random.choice(['Asia/Shanghai', 'UTC', 'America/New_York', 'Europe/London']),
            'language': random.choice(['zh-CN', 'en-US', 'zh-TW', 'en-GB']),
        }
    
    def generate_behavior_sequence(self, is_normal=True):
        """生成行为序列（增加重叠度，更难区分）"""
        if is_normal:
            # 正常用户：增加一些边界情况，与羊毛党有重叠
            if random.random() < 0.2:  # 20%概率生成"急躁"的正常用户
                return {
                    'page_stay_time': random.uniform(2, 6),  # 稍微短一点
                    'click_count': random.randint(2, 5),
                    'scroll_count': random.randint(2, 8),
                    'path_entropy': random.uniform(0.5, 0.8),  # 中等熵
                    'mouse_trajectory_entropy': random.uniform(0.6, 0.9),
                    'request_frequency': random.uniform(1.5, 3.0),  # 稍高频率
                }
            else:
                # 标准正常用户
                return {
                    'page_stay_time': random.uniform(3, 15),
                    'click_count': random.randint(3, 10),
                    'scroll_count': random.randint(5, 20),
                    'path_entropy': random.uniform(0.7, 1.0),
                    'mouse_trajectory_entropy': random.uniform(0.8, 1.0),
                    'request_frequency': random.uniform(0.5, 2.0),
                }
        else:
            # 羊毛党：增加一些"伪装"情况，让部分行为接近正常
            if random.random() < 0.3:  # 30%概率生成"高级"羊毛党（行为更接近正常）
                return {
                    'page_stay_time': random.uniform(3, 7),  # 接近正常
                    'click_count': random.randint(3, 6),  # 接近正常
                    'scroll_count': random.randint(4, 10),  # 接近正常
                    'path_entropy': random.uniform(0.4, 0.7),  # 中等偏低
                    'mouse_trajectory_entropy': random.uniform(0.3, 0.6),  # 偏低但不太明显
                    'request_frequency': random.uniform(1.5, 3.5),  # 稍高但不太明显
                }
            else:
                # 典型羊毛党
                return {
                    'page_stay_time': random.uniform(1, 5),
                    'click_count': random.randint(1, 3),
                    'scroll_count': random.randint(0, 5),
                    'path_entropy': random.uniform(0.1, 0.5),
                    'mouse_trajectory_entropy': random.uniform(0.1, 0.4),
                    'request_frequency': random.uniform(2.0, 5.0),
                }
