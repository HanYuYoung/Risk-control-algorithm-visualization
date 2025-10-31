"""
模型训练脚本 - 使用网格搜索和交叉验证训练多个模型
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from itertools import product
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm
from config import MODEL_CONFIG, DATA_DIR
from models.feature_extractor import FeatureExtractor

def load_and_extract_features(data_file=None):
    """从CSV文件加载原始数据并提取特征"""
    if data_file is None:
        data_file = os.path.join(DATA_DIR, 'training_data.csv')
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            f"训练数据文件不存在: {data_file}\n"
            f"请先运行: python generate_data.py"
        )
    
    print(f"加载训练数据: {data_file}")
    df = pd.read_csv(data_file)
    
    print(f"  原始样本数: {len(df)}")
    print(f"  正常样本: {np.sum(df['label']==0)}")
    print(f"  异常样本: {np.sum(df['label']==1)}")
    
    # 提取特征
    print("\n提取特征...")
    extractor = FeatureExtractor()
    X = []
    y = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="特征提取", unit="样本"):
        # 重构请求数据格式
        request_data = {
            'phone': str(row['phone']),
            'ip': str(row['ip']),
            'device_id': str(row['device_id']),
            'user_agent': str(row.get('user_agent', '')),
            'timestamp': float(row['timestamp']),
            'device_fingerprint': {
                'canvas_fingerprint': str(row.get('canvas_fingerprint', '')),
                'fonts': str(row.get('fonts', '')),
                'screen_resolution': str(row.get('screen_resolution', '1920x1080')),
                'timezone': str(row.get('timezone', 'Asia/Shanghai')),
                'language': str(row.get('language', 'zh-CN')),
            },
            'behavior': {
                'page_stay_time': float(row['page_stay_time']),
                'click_count': int(row['click_count']),
                'scroll_count': int(row['scroll_count']),
                'path_entropy': float(row['path_entropy']),
                'mouse_trajectory_entropy': float(row.get('mouse_trajectory_entropy', 0.5)),
                'request_frequency': float(row.get('request_frequency', 1.0)),
            }
        }
        
        features = extractor.extract_features(request_data)
        X.append(features)
        y.append(int(row['label']))
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n[OK] 特征提取完成")
    print(f"  特征维度: {X.shape[1]}")
    
    return X, y

def train_xgboost(X, y, model_path):
    """使用网格搜索训练XGBoost模型（不使用交叉验证）"""
    print("\n" + "="*60)
    print("训练XGBoost模型（网格搜索）")
    print("="*60)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    print(f"开始网格搜索（共 {len(param_combinations)} 种参数组合）...")
    
    best_score = -1
    best_params = None
    best_model = None
    
    # 手动网格搜索
    for params in tqdm(param_combinations, desc="网格搜索"):
        param_dict = dict(zip(param_names, params))
        model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1,
            **param_dict
        )
        model.fit(X_train, y_train)
        
        # 在验证集上评估
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred_proba)
        
        if val_auc > best_score:
            best_score = val_auc
            best_params = param_dict
            best_model = model
    
    # 评估最佳模型
    train_score = best_model.score(X_train, y_train)
    val_score = best_model.score(X_val, y_val)
    val_pred_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred_proba)
    
    print(f"\n[OK] 训练完成")
    print(f"  最佳参数: {best_params}")
    print(f"  最佳AUC得分: {best_score:.4f}")
    print(f"  训练集准确率: {train_score:.4f}")
    print(f"  验证集准确率: {val_score:.4f}")
    print(f"  验证集AUC: {val_auc:.4f}")
    
    joblib.dump(best_model, model_path)
    print(f"[OK] 模型已保存: {model_path}")
    
    return best_model

def train_random_forest(X, y, model_path):
    """使用网格搜索训练RandomForest模型（不使用交叉验证）"""
    print("\n" + "="*60)
    print("训练RandomForest模型（网格搜索）")
    print("="*60)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    print(f"开始网格搜索（共 {len(param_combinations)} 种参数组合）...")
    
    best_score = -1
    best_params = None
    best_model = None
    
    # 手动网格搜索
    for params in tqdm(param_combinations, desc="网格搜索"):
        param_dict = dict(zip(param_names, params))
        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **param_dict
        )
        model.fit(X_train, y_train)
        
        # 在验证集上评估
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred_proba)
        
        if val_auc > best_score:
            best_score = val_auc
            best_params = param_dict
            best_model = model
    
    train_score = best_model.score(X_train, y_train)
    val_score = best_model.score(X_val, y_val)
    val_pred_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred_proba)
    
    print(f"\n[OK] 训练完成")
    print(f"  最佳参数: {best_params}")
    print(f"  最佳AUC得分: {best_score:.4f}")
    print(f"  训练集准确率: {train_score:.4f}")
    print(f"  验证集准确率: {val_score:.4f}")
    print(f"  验证集AUC: {val_auc:.4f}")
    
    joblib.dump(best_model, model_path)
    print(f"[OK] 模型已保存: {model_path}")
    
    return best_model

def train_gbdt(X, y, model_path):
    """使用网格搜索训练GBDT模型（不使用交叉验证）"""
    print("\n" + "="*60)
    print("训练GBDT模型（网格搜索）")
    print("="*60)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9]
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    print(f"开始网格搜索（共 {len(param_combinations)} 种参数组合）...")
    
    best_score = -1
    best_params = None
    best_model = None
    
    # 手动网格搜索
    for params in tqdm(param_combinations, desc="网格搜索"):
        param_dict = dict(zip(param_names, params))
        model = GradientBoostingClassifier(
            random_state=42,
            **param_dict
        )
        model.fit(X_train, y_train)
        
        # 在验证集上评估
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred_proba)
        
        if val_auc > best_score:
            best_score = val_auc
            best_params = param_dict
            best_model = model
    
    train_score = best_model.score(X_train, y_train)
    val_score = best_model.score(X_val, y_val)
    val_pred_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred_proba)
    
    print(f"\n[OK] 训练完成")
    print(f"  最佳参数: {best_params}")
    print(f"  最佳AUC得分: {best_score:.4f}")
    print(f"  训练集准确率: {train_score:.4f}")
    print(f"  验证集准确率: {val_score:.4f}")
    print(f"  验证集AUC: {val_auc:.4f}")
    
    joblib.dump(best_model, model_path)
    print(f"[OK] 模型已保存: {model_path}")
    
    return best_model

def train_lightgbm(X, y, model_path):
    """使用网格搜索训练LightGBM模型（不使用交叉验证）"""
    print("\n" + "="*60)
    print("训练LightGBM模型（网格搜索）")
    print("="*60)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'num_leaves': [31, 50, 70],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9]
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    print(f"开始网格搜索（共 {len(param_combinations)} 种参数组合）...")
    
    best_score = -1
    best_params = None
    best_model = None
    
    # 手动网格搜索
    for params in tqdm(param_combinations, desc="网格搜索"):
        param_dict = dict(zip(param_names, params))
        model = lgb.LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **param_dict
        )
        model.fit(X_train, y_train)
        
        # 在验证集上评估
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred_proba)
        
        if val_auc > best_score:
            best_score = val_auc
            best_params = param_dict
            best_model = model
    
    train_score = best_model.score(X_train, y_train)
    val_score = best_model.score(X_val, y_val)
    val_pred_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred_proba)
    
    print(f"\n[OK] 训练完成")
    print(f"  最佳参数: {best_params}")
    print(f"  最佳AUC得分: {best_score:.4f}")
    print(f"  训练集准确率: {train_score:.4f}")
    print(f"  验证集准确率: {val_score:.4f}")
    print(f"  验证集AUC: {val_auc:.4f}")
    
    joblib.dump(best_model, model_path)
    print(f"[OK] 模型已保存: {model_path}")
    
    return best_model

def train_isolation_forest(X, model_path, contamination=0.2):
    """训练IsolationForest模型"""
    print("\n" + "="*60)
    print("训练IsolationForest模型（无监督）")
    print("="*60)
    
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X)
    
    predictions = model.predict(X)
    anomaly_ratio = (predictions == -1).sum() / len(predictions)
    
    print(f"[OK] 训练完成")
    print(f"  异常样本比例: {anomaly_ratio:.4f}")
    
    joblib.dump(model, model_path)
    print(f"[OK] 模型已保存: {model_path}")
    
    return model

def main():
    """主训练函数"""
    print("=" * 60)
    print("风控模型训练（多模型 + 网格搜索）")
    print("=" * 60)
    
    # 加载数据并提取特征
    X, y = load_and_extract_features()
    
    # 训练所有监督学习模型
    print("\n开始训练监督学习模型...")
    train_xgboost(X, y, MODEL_CONFIG['xgb_model_path'])
    train_random_forest(X, y, MODEL_CONFIG['rf_model_path'])
    train_gbdt(X, y, MODEL_CONFIG['gbdt_model_path'])
    train_lightgbm(X, y, MODEL_CONFIG['lgb_model_path'])
    
    # 训练无监督模型（使用正常样本）
    print("\n开始训练无监督模型...")
    train_isolation_forest(X[y == 0], MODEL_CONFIG['isolation_forest_path'])
    
    print("\n" + "=" * 60)
    print("[OK] 所有模型训练完成")
    print("=" * 60)
    print("\n模型文件保存在:")
    print(f"  - XGBoost: {MODEL_CONFIG['xgb_model_path']}")
    print(f"  - RandomForest: {MODEL_CONFIG['rf_model_path']}")
    print(f"  - GBDT: {MODEL_CONFIG['gbdt_model_path']}")
    print(f"  - LightGBM: {MODEL_CONFIG['lgb_model_path']}")
    print(f"  - IsolationForest: {MODEL_CONFIG['isolation_forest_path']}")

if __name__ == '__main__':
    main()
