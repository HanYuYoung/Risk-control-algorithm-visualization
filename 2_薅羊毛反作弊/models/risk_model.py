"""
风控模型 - AI驱动的风险检测（支持多模型选择）
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from sklearn.ensemble import IsolationForest
import json
from datetime import datetime
from config import MODEL_CONFIG

class RiskModel:
    """风控模型"""
    
    def __init__(self):
        self.models = {
            'xgb': None,
            'rf': None,
            'gbdt': None,
            'lgb': None
        }
        self.isolation_forest = None
        self.feature_extractor = None
        self.current_model_type = MODEL_CONFIG.get('default_model', 'xgb')
        self.ensemble_weights = MODEL_CONFIG.get('ensemble_weights', {'supervised': 0.6, 'unsupervised': 0.4})
        self.is_trained = False
    
    def load_models(self, model_type=None):
        """加载模型
        
        Args:
            model_type: 要加载的模型类型，None表示加载所有可用模型
        """
        try:
            from models.feature_extractor import FeatureExtractor
        except ImportError:
            import sys as _sys
            import os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
            from models.feature_extractor import FeatureExtractor
        
        self.feature_extractor = FeatureExtractor()
        
        # 加载所有可用的监督学习模型
        model_paths = {
            'xgb': MODEL_CONFIG['xgb_model_path'],
            'rf': MODEL_CONFIG['rf_model_path'],
            'gbdt': MODEL_CONFIG['gbdt_model_path'],
            'lgb': MODEL_CONFIG['lgb_model_path']
        }
        
        loaded_models = []
        for model_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                try:
                    self.models[model_name] = joblib.load(model_path)
                    loaded_models.append(model_name)
                    print(f"[OK] 已加载{model_name.upper()}模型: {model_path}")
                except Exception as e:
                    print(f"[WARN] 无法加载{model_name.upper()}模型: {e}")
        
        # 加载无监督模型
        iso_path = MODEL_CONFIG['isolation_forest_path']
        if os.path.exists(iso_path):
            try:
                self.isolation_forest = joblib.load(iso_path)
                print(f"[OK] 已加载IsolationForest模型: {iso_path}")
            except Exception as e:
                print(f"[WARN] 无法加载IsolationForest模型: {e}")
        
        # 设置当前使用的模型
        if model_type and model_type in self.models and self.models[model_type] is not None:
            self.current_model_type = model_type
        elif loaded_models:
            # 使用第一个加载成功的模型
            self.current_model_type = loaded_models[0]
            if model_type and model_type not in loaded_models:
                print(f"[WARN] 请求的模型 {model_type} 未加载，使用默认模型: {self.current_model_type}")
        
        self.is_trained = len(loaded_models) > 0
    
    def set_model(self, model_type):
        """设置当前使用的模型
        
        Args:
            model_type: 'xgb', 'rf', 'gbdt', 'lgb'
        """
        if model_type in self.models and self.models[model_type] is not None:
            self.current_model_type = model_type
            return True
        return False
    
    def get_available_models(self):
        """获取已加载的模型列表"""
        return [name for name, model in self.models.items() if model is not None]
    
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
        current_model = self.models[self.current_model_type]
        if current_model:
            try:
                supervised_score = float(current_model.predict_proba(features_2d)[0][1])
            except:
                supervised_score = 0.5
        
        # 无监督模型预测
        unsupervised_score = 0.0
        if self.isolation_forest:
            try:
                anomaly_score = self.isolation_forest.decision_function(features_2d)[0]
                # 将异常分数转换为0-1风险分数
                # IsolationForest返回值：负值表示异常，正值表示正常
                # 归一化到0-1范围
                unsupervised_score = max(0, min(1, (1 - anomaly_score) / 2))
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
                'model_type': self.current_model_type,
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
                elif len(features) > 5 and features[5] < 0.5:
                    risk_type = '行为异常'
            return 'review', risk_type
        else:
            # 高风险
            risk_type = '批量操作+环境异常'
            if features is not None:
                if features[0] > 5 and features[1] > 3:
                    risk_type = 'IP聚集+设备聚集'
                elif len(features) > 6 and features[6] == 1:  # 黑名单手机号
                    risk_type = '垃圾账号'
            return 'reject', risk_type
