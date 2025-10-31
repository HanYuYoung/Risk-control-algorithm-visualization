"""
特征提取器 - 增强版（40+维度特征）
"""
import time
import hashlib
import random
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

class FeatureExtractor:
    """特征提取器 - 增强版"""
    
    def __init__(self):
        self.ip_registry = defaultdict(list)
        self.device_registry = defaultdict(list)
        self.phone_registry = defaultdict(int)
        self.ip_device_mapping = defaultdict(set)  # IP-设备关联
        self.device_phone_mapping = defaultdict(set)  # 设备-手机号关联
        self.blacklist_phones = set()
        self._init_blacklist()
    
    def _init_blacklist(self):
        """初始化黑名单"""
        for i in range(100):
            phone = f"199{random.randint(10000000, 99999999)}"
            self.blacklist_phones.add(phone)
    
    def extract_features(self, request_data):
        """
        提取增强特征向量（40+维度）
        
        Args:
            request_data: 请求数据字典
            
        Returns:
            numpy array: 特征向量
        """
        phone = request_data.get('phone', '')
        ip = request_data.get('ip', '')
        device_id = request_data.get('device_id', '')
        timestamp = request_data.get('timestamp', time.time())
        behavior = request_data.get('behavior', {})
        device_fp = request_data.get('device_fingerprint', {})
        user_agent = request_data.get('user_agent', '')
        
        current_time = timestamp
        time_window = 3600  # 1小时窗口
        
        dt = datetime.fromtimestamp(timestamp)
        
        # ========== 1. 聚集度特征（8维）==========
        ip_reg_count = self._get_recent_count(self.ip_registry, ip, current_time, time_window)
        device_reg_count = self._get_recent_count(self.device_registry, device_id, current_time, time_window)
        ip_device_count = len(self.ip_device_mapping.get(ip, set()))  # IP关联的设备数
        device_ip_count = len([dip for dip, devices in self.ip_device_mapping.items() if device_id in devices])  # 设备关联的IP数
        phone_device_count = len(self.device_phone_mapping.get(device_id, set()))  # 设备关联的手机号数
        ip_phone_count = sum(1 for dip, devices in self.ip_device_mapping.items() 
                           if ip == dip and phone in [p for d, phones in self.device_phone_mapping.items() 
                                                      for p in phones if d in devices])  # IP关联的手机号数
        same_ip_different_devices = max(0, ip_device_count - 1)  # 同一IP不同设备数
        same_device_different_phones = max(0, phone_device_count - 1)  # 同一设备不同手机号数
        
        # ========== 2. 行为特征（12维）==========
        page_stay_time = behavior.get('page_stay_time', 0)
        click_count = behavior.get('click_count', 0)
        scroll_count = behavior.get('scroll_count', 0)
        path_entropy = behavior.get('path_entropy', 0)
        mouse_trajectory_entropy = behavior.get('mouse_trajectory_entropy', 0)
        request_frequency = behavior.get('request_frequency', 0)
        
        # 行为比例特征
        clicks_per_second = click_count / max(page_stay_time, 0.1)  # 点击频率
        scrolls_per_second = scroll_count / max(page_stay_time, 0.1)  # 滚动频率
        clicks_per_scroll = click_count / max(scroll_count, 1)  # 点击/滚动比
        behavior_diversity = (path_entropy + mouse_trajectory_entropy) / 2  # 行为多样性
        
        # 行为异常标记
        short_stay = 1 if page_stay_time < 3 else 0
        low_interaction = 1 if (click_count < 3 and scroll_count < 3) else 0
        high_frequency = 1 if request_frequency > 2.0 else 0
        
        # ========== 3. 账号特征（5维）==========
        phone_in_blacklist = 1 if phone in self.blacklist_phones else 0
        phone_history_count = self.phone_registry.get(phone, 0)
        phone_segment = int(phone[:3]) if len(phone) >= 3 else 0  # 手机号段
        phone_is_virtual = 1 if phone.startswith(('17', '19')) else 0  # 虚拟号段
        phone_reg_time_span = 0  # 同一手机号注册时间跨度（简化）
        
        # ========== 4. 时间特征（6维）==========
        hour = dt.hour
        minute = dt.minute
        day_of_week = dt.weekday()  # 0=Monday
        is_workday = 1 if day_of_week < 5 else 0  # 工作日
        is_peak_hour = 1 if 9 <= hour <= 11 or 14 <= hour <= 17 else 0  # 高峰时段
        is_night = 1 if 22 <= hour or hour < 6 else 0  # 深夜时段
        
        # ========== 5. 设备指纹特征（8维）==========
        device_hash = int(hashlib.md5(device_id.encode()).hexdigest()[:8], 16)
        ip_hash = int(hashlib.md5(ip.encode()).hexdigest()[:8], 16)
        canvas_hash = int(hashlib.md5(device_fp.get('canvas_fingerprint', '').encode()).hexdigest()[:8], 16) if device_fp.get('canvas_fingerprint') else 0
        
        # User-Agent特征
        is_mobile = 1 if 'Mobile' in user_agent or 'Android' in user_agent or 'iPhone' in user_agent else 0
        is_chrome = 1 if 'Chrome' in user_agent else 0
        is_safari = 1 if 'Safari' in user_agent and 'Chrome' not in user_agent else 0
        
        # 设备指纹稳定性
        screen_w, screen_h = map(int, device_fp.get('screen_resolution', '1920x1080').split('x')) if 'x' in device_fp.get('screen_resolution', '') else (1920, 1080)
        screen_area = screen_w * screen_h
        
        # ========== 6. 网络特征（4维）==========
        # 模拟网络延迟（基于IP段）
        ip_parts = ip.split('.') if '.' in ip else ['0', '0', '0', '0']
        network_segment = int(ip_parts[0]) if ip_parts else 0
        estimated_latency = network_segment % 100  # 模拟延迟(ms)
        
        # 是否为代理IP（简化判断）
        is_proxy_likely = 1 if ip_reg_count > 3 else 0
        ip_reputation_score = 100 - min(ip_reg_count * 10, 100)  # IP信誉分数
        
        # ========== 7. 关联特征（4维）==========
        device_fingerprint_uniqueness = abs(device_hash % 100) / 100  # 设备指纹唯一性
        ip_geolocation_consistency = 1 if network_segment < 50 else 0  # IP地理位置一致性（简化）
        registration_pattern_anomaly = 1 if (ip_reg_count > 5 or device_reg_count > 3) else 0  # 注册模式异常
        cluster_score = (ip_reg_count + device_reg_count) / 10  # 聚集分数
        
        # ========== 8. 组合特征（5维）==========
        risk_score_base = (
            float(ip_reg_count > 5) * 0.3 +
            float(device_reg_count > 3) * 0.2 +
            float(page_stay_time < 3) * 0.15 +
            float(path_entropy < 0.5) * 0.15 +
            float(phone_in_blacklist) * 0.2
        )
        
        behavior_anomaly_score = (
            float(short_stay) * 0.3 +
            float(low_interaction) * 0.3 +
            float(high_frequency) * 0.2 +
            float(behavior_diversity < 0.5) * 0.2
        )
        
        device_anomaly_score = (
            float(device_reg_count > 3) * 0.4 +
            float(same_device_different_phones > 2) * 0.3 +
            float(is_proxy_likely) * 0.3
        )
        
        timing_anomaly_score = (
            float(is_night) * 0.3 +
            float(not is_workday) * 0.2 +
            float(request_frequency > 3) * 0.5
        )
        
        comprehensive_risk_score = (
            risk_score_base * 0.3 +
            behavior_anomaly_score * 0.3 +
            device_anomaly_score * 0.2 +
            timing_anomaly_score * 0.2
        )
        
        # 构建特征向量（42维）
        features = np.array([
            # 聚集度特征（8）
            ip_reg_count,
            device_reg_count,
            ip_device_count,
            device_ip_count,
            phone_device_count,
            ip_phone_count,
            same_ip_different_devices,
            same_device_different_phones,
            
            # 行为特征（12）
            page_stay_time,
            click_count,
            scroll_count,
            path_entropy,
            mouse_trajectory_entropy,
            request_frequency,
            clicks_per_second,
            scrolls_per_second,
            clicks_per_scroll,
            behavior_diversity,
            short_stay,
            low_interaction,
            high_frequency,
            
            # 账号特征（5）
            phone_in_blacklist,
            phone_history_count,
            phone_segment % 1000,
            phone_is_virtual,
            phone_reg_time_span,
            
            # 时间特征（6）
            hour,
            minute,
            day_of_week,
            is_workday,
            is_peak_hour,
            is_night,
            
            # 设备指纹特征（8）
            device_hash % 1000,
            ip_hash % 1000,
            canvas_hash % 1000,
            is_mobile,
            is_chrome,
            is_safari,
            screen_area / 10000,  # 归一化
            device_fingerprint_uniqueness,
            
            # 网络特征（4）
            estimated_latency,
            is_proxy_likely,
            ip_reputation_score / 100,
            network_segment,
            
            # 关联特征（4）
            ip_geolocation_consistency,
            registration_pattern_anomaly,
            cluster_score,
            float(ip_device_count > 3),  # 多设备标记
            
            # 组合特征（5）
            risk_score_base,
            behavior_anomaly_score,
            device_anomaly_score,
            timing_anomaly_score,
            comprehensive_risk_score,
        ])
        
        # 更新注册历史
        self.ip_registry[ip].append(current_time)
        self.device_registry[device_id].append(current_time)
        self.phone_registry[phone] += 1
        self.ip_device_mapping[ip].add(device_id)
        self.device_phone_mapping[device_id].add(phone)
        
        return features
    
    def _get_recent_count(self, registry, key, current_time, window):
        """获取时间窗口内的注册次数"""
        if key not in registry:
            return 0
        window_start = current_time - window
        recent_times = [t for t in registry[key] if t >= window_start]
        return len(recent_times)
