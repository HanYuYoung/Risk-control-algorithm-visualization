"""
风控模型 - AI驱动的风险检测
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from sklearn.ensemble import IsolationForest
import json
from datetime import datetime

class RiskModel:
    """风控模型"""
    
    def __init__(self):
        self.xgb_model = None
        self.lstm_model = None
        self.isolation_forest = None
        self.feature_extractor = None
        self.ensemble_weights = {'supervised': 0.6, 'unsupervised': 0.4}
        self.is_trained = False
    
    def load_models(self, xgb_path=None, isolation_forest_path=None):
        """加载模型"""
        try:
            from models.feature_extractor import FeatureExtractor
        except ImportError:
            import sys as _sys
            import os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from models.feature_extractor import FeatureExtractor
        
        self.feature_extractor = FeatureExtractor()
        
        # 如果模型文件存在，加载它们
        if xgb_path and os.path.exists(xgb_path):
            try:
                self.xgb_model = joblib.load(xgb_path)
                print(f"[OK] 已加载XGBoost模型: {xgb_path}")
            except:
                print(f"[WARN] 无法加载XGBoost模型: {xgb_path}")
        
        if isolation_forest_path and os.path.exists(isolation_forest_path):
            try:
                self.isolation_forest = joblib.load(isolation_forest_path)
                print(f"[OK] 已加载IsolationForest模型: {isolation_forest_path}")
            except:
                print(f"[WARN] 无法加载IsolationForest模型: {isolation_forest_path}")
        
        self.is_trained = (self.xgb_model is not None) and (self.isolation_forest is not None)
    
    def predict(self, request_data):
        """
        预测风险分数
        
        Args:
            request_data: 请求数据
            
        Returns:
            dict: {
                'risk_score': float,  # 风险分数 0-1
                'risk_level': str,    # pass/review/reject
                'risk_type': str,     # 风险类型
                'details': dict       # 详细信息
            }
        """
        if not self.is_trained:
            # 如果模型未训练，使用简单规则
            return self._simple_rule_based_prediction(request_data)
        
        # 提取特征
        features = self.feature_extractor.extract_features(request_data)
        features_2d = features.reshape(1, -1)
        
        # 监督模型预测
        supervised_score = 0.0
        if self.xgb_model:
            try:
                supervised_score = float(self.xgb_model.predict_proba(features_2d)[0][1])
            except:
                supervised_score = 0.5
        
        # 无监督模型预测
        unsupervised_score = 0.0
        if self.isolation_forest:
            try:
                anomaly_score = self.isolation_forest.decision_function(features_2d)[0]
                # 将异常分数转换为0-1风险分数
                # IsolationForest返回值：-1表示异常，1表示正常
                unsupervised_score = max(0, (1 - anomaly_score) / 2)
            except:
                unsupervised_score = 0.5
        
        # 模型融合
        risk_score = (
            self.ensemble_weights['supervised'] * supervised_score +
            self.ensemble_weights['unsupervised'] * unsupervised_score
        )
        
        # 确定风险等级和类型
        risk_level, risk_type = self._determine_risk_level(risk_score, features, request_data)
        
        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'risk_type': risk_type,
            'details': {
                'supervised_score': float(supervised_score),
                'unsupervised_score': float(unsupervised_score),
                'features': features.tolist(),
            }
        }
    
    def _simple_rule_based_prediction(self, request_data):
        """简单基于规则的预测（模型未训练时使用）"""
        behavior = request_data.get('behavior', {})
        ip = request_data.get('ip', '')
        phone = request_data.get('phone', '')
        
        risk_score = 0.0
        risk_type = '无'
        
        # 简单规则
        if behavior.get('page_stay_time', 10) < 3:
            risk_score += 0.3
        
        if behavior.get('path_entropy', 1.0) < 0.5:
            risk_score += 0.2
        
        if behavior.get('click_count', 5) < 3:
            risk_score += 0.2
        
        # IP/设备聚集检测（需要从feature_extractor获取）
        if self.feature_extractor:
            features = self.feature_extractor.extract_features(request_data)
            if features[0] > 5:  # IP注册次数 > 5
                risk_score += 0.3
                risk_type = 'IP聚集'
            if features[1] > 3:  # 设备注册次数 > 3
                risk_score += 0.2
                risk_type = '设备聚集' if not risk_type else risk_type + '+设备聚集'
        
        risk_score = min(1.0, risk_score)
        risk_level, _ = self._determine_risk_level(risk_score, None, request_data)
        
        return {
            'risk_score': float(risk_score),
            'risk_level': risk_level,
            'risk_type': risk_type or '批量操作',
            'details': {}
        }
    
    def _determine_risk_level(self, risk_score, features, request_data):
        """确定风险等级"""
        if risk_score < 0.5:
            return 'pass', '无'
        elif risk_score < 0.8:
            # 判断风险类型
            risk_type = '批量操作'
            if features is not None:
                if features[0] > 5:
                    risk_type = 'IP聚集'
                elif features[1] > 3:
                    risk_type = '设备聚集'
                elif features[5] < 0.5:
                    risk_type = '行为异常'
            return 'review', risk_type
        else:
            # 高风险
            risk_type = '批量操作+环境异常'
            if features is not None:
                if features[0] > 5 and features[1] > 3:
                    risk_type = 'IP聚集+设备聚集'
                elif features[6] == 1:  # 黑名单手机号
                    risk_type = '垃圾账号'
            return 'reject', risk_type

