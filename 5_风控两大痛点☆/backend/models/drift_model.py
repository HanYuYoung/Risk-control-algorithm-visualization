"""
漂移检测和模型重训练
"""
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import sys

# 添加路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.utils import (
    preprocess_data, load_prosper_data, 
    detect_feature_drift, save_model, load_model
)


def simulate_drift_and_retrain(drift_strength=0.3):
    """
    模拟特征漂移并重训练模型
    """
    # 加载原始训练数据
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(backend_dir, 'models')
    base_train_data = load_model(os.path.join(model_dir, 'base_train_data.pkl'))
    X_train_base = base_train_data['X_train']
    feature_names = base_train_data['feature_names']
    
    # 加载基础模型
    base_model = load_model(os.path.join(model_dir, 'base_model.pkl'))
    
    # 生成漂移后的新数据
    df = load_prosper_data()
    X_drift, y_drift, _ = preprocess_data(df, is_drift=True, drift_strength=drift_strength)
    
    # 检测特征漂移
    drift_results, drifted_features = detect_feature_drift(
        X_train_base, X_drift, feature_names
    )
    
    # 用原始模型预测漂移数据（模拟旧模型失效）
    # 处理不同类型的模型对象
    if hasattr(base_model, 'predict_proba'):
        # sklearn接口的模型（LightGBM, XGBoost, Random Forest, CatBoost等）
        y_pred_proba_old = base_model.predict_proba(X_drift)[:, 1]
    elif hasattr(base_model, 'predict'):
        # 原生LightGBM Booster模型
        y_pred_proba_old = base_model.predict(X_drift, num_iteration=base_model.best_iteration if hasattr(base_model, 'best_iteration') else None)
    else:
        raise ValueError(f"不支持的模型类型: {type(base_model)}")
    auc_old = roc_auc_score(y_drift, y_pred_proba_old)
    
    # 重训练模型（使用漂移后的数据）
    train_data = lgb.Dataset(X_drift, label=y_drift)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    new_model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        callbacks=[lgb.log_evaluation(0)]
    )
    
    # 用新模型预测
    y_pred_proba_new = new_model.predict(X_drift, num_iteration=new_model.best_iteration)
    auc_new = roc_auc_score(y_drift, y_pred_proba_new)
    
    # 保存新模型
    drift_model_path = os.path.join(model_dir, 'drift_model.pkl')
    save_model(new_model, drift_model_path)
    
    return {
        'drift_results': drift_results,
        'drifted_features': drifted_features,
        'auc_old': auc_old,
        'auc_new': auc_new,
        'X_drift': X_drift,
        'y_drift': y_drift,
        'y_pred_proba_old': y_pred_proba_old,
        'y_pred_proba_new': y_pred_proba_new,
        'new_model': new_model
    }

