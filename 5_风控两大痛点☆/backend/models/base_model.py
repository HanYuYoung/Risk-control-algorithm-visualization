"""
初始模型训练 - 支持多模型选择、K折交叉验证和超参数调优
"""
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, make_scorer
import numpy as np
import os
import sys

# 获取CPU核心数用于多线程
try:
    import multiprocessing
    CPU_COUNT = multiprocessing.cpu_count()
except:
    CPU_COUNT = 4  # 默认值

# 尝试导入可选库
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# 添加路径以便导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.utils import preprocess_data, load_prosper_data, save_model


def get_model_config(model_name):
    """获取不同模型的配置"""
    configs = {
        'LightGBM': {
            'class': lgb.LGBMClassifier,
            'default_params': {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42,
                'n_estimators': 100,
                'n_jobs': -1  # 使用所有CPU核心
            },
            'param_grid': {
                'num_leaves': [31, 50, 70],
                'learning_rate': [0.05, 0.1, 0.15],
                'feature_fraction': [0.8, 0.9, 1.0],
                'bagging_fraction': [0.7, 0.8, 0.9],
                'max_depth': [5, 7, 9]
            },
            'param_ranges': {
                'num_leaves': (31, 100),
                'learning_rate': (0.01, 0.2),
                'feature_fraction': (0.6, 1.0),
                'bagging_fraction': (0.6, 1.0),
                'max_depth': (3, 12)
            }
        },
        'XGBoost': {
            'class': xgb.XGBClassifier if XGBOOST_AVAILABLE else None,
            'default_params': {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'random_state': 42,
                'n_estimators': 100,
                'verbosity': 0,
                'n_jobs': -1  # 使用所有CPU核心
            },
            'param_grid': {
                'max_depth': [5, 7, 9],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5]
            },
            'param_ranges': {
                'max_depth': (3, 12),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'min_child_weight': (1, 10)
            }
        },
        'Random Forest': {
            'class': RandomForestClassifier,
            'default_params': {
                'n_estimators': 100,
                'random_state': 42,
                'n_jobs': -1
            },
            'param_grid': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'param_ranges': {
                'n_estimators': (50, 500),
                'max_depth': (5, 30),
                'min_samples_split': (2, 20),
                'min_samples_leaf': (1, 10),
                'max_features': ['sqrt', 'log2', None]  # 类别参数
            }
        },
        'CatBoost': {
            'class': CatBoostClassifier if CATBOOST_AVAILABLE else None,
            'default_params': {
                'objective': 'Logloss',
                'eval_metric': 'AUC',
                'random_state': 42,
                'iterations': 100,
                'verbose': False,
                'thread_count': -1  # 使用所有CPU核心
            },
            'param_grid': {
                'depth': [5, 7, 9],
                'learning_rate': [0.05, 0.1, 0.15],
                'l2_leaf_reg': [1, 3, 5],
                'iterations': [100, 200]
            },
            'param_ranges': {
                'depth': (4, 10),
                'learning_rate': (0.01, 0.3),
                'l2_leaf_reg': (1, 10),
                'iterations': (50, 300)
            }
        }
    }
    return configs.get(model_name)


def bayesian_optimization(X_train, y_train, ModelClass, default_params, param_ranges, cv_folds, n_trials=50, progress_callback=None):
    """
    使用Optuna进行贝叶斯优化
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna未安装，请运行: pip install optuna")
    
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    def objective(trial):
        # 根据参数范围建议参数
        params = default_params.copy()
        for param_name, param_range in param_ranges.items():
            if isinstance(param_range, list):
                # 类别参数
                params[param_name] = trial.suggest_categorical(param_name, param_range)
            elif isinstance(param_range, tuple) and len(param_range) == 2:
                # 数值参数（整数或浮点数）
                if param_name in ['num_leaves', 'max_depth', 'n_estimators', 'iterations', 
                                 'min_samples_split', 'min_samples_leaf', 'depth', 'l2_leaf_reg']:
                    # 整数参数
                    params[param_name] = trial.suggest_int(param_name, int(param_range[0]), int(param_range[1]))
                else:
                    # 浮点数参数
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
        
        # K折交叉验证
        cv_scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_train_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_train_fold = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            y_val_fold = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
            
            model = ModelClass(**params)
            model.fit(X_train_fold, y_train_fold)
            y_val_pred = model.predict_proba(X_val_fold)[:, 1]
            fold_auc = roc_auc_score(y_val_fold, y_val_pred)
            cv_scores.append(fold_auc)
        
        return np.mean(cv_scores)
    
    # 创建study并优化
    study = optuna.create_study(direction='maximize', study_name=f'{ModelClass.__name__}_optimization')
    
    # 添加进度回调
    def progress_callback_wrapper(study, trial):
        if progress_callback:
            progress = 20 + int(60 * (trial.number + 1) / n_trials)
            progress_callback(progress, 100, f"贝叶斯优化进度: {trial.number + 1}/{n_trials} 次试验 (当前最佳AUC: {study.best_value:.4f})")
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, callbacks=[progress_callback_wrapper] if progress_callback else None)
    
    # 获取最佳参数
    best_params = default_params.copy()
    best_params.update(study.best_params)
    
    return best_params, study.best_value, study


def train_base_model(model_name='LightGBM', use_cv=True, cv_folds=5, tune_hyperparams=True, 
                     optimization_method='bayesian', n_trials=50, handle_imbalance=True,
                     imbalance_method='class_weight', progress_callback=None):
    """
    训练初始模型
    
    参数:
        model_name: 模型名称 ('LightGBM', 'XGBoost', 'Random Forest', 'CatBoost')
        use_cv: 是否使用K折交叉验证
        cv_folds: K折交叉验证的折数
        tune_hyperparams: 是否进行超参数调优
        optimization_method: 优化方法 ('grid' 或 'bayesian')
        n_trials: 贝叶斯优化的试验次数（仅当optimization_method='bayesian'时使用）
        handle_imbalance: 是否处理样本不平衡
        imbalance_method: 不平衡处理方法 ('class_weight', 'smote', 'adasyn')
        progress_callback: 进度回调函数 (current, total, message)
    
    返回:
        包含模型、评估指标和训练信息的字典
    """
    # 检查模型是否可用
    config = get_model_config(model_name)
    if config is None or config['class'] is None:
        raise ValueError(f"模型 {model_name} 不可用，请检查是否已安装相应库")
    
    # 加载数据
    if progress_callback:
        progress_callback(0, 100, "正在加载数据...")
    df = load_prosper_data()
    X, y, feature_names = preprocess_data(df, is_drift=False)
    
    # 划分训练集和测试集
    if progress_callback:
        progress_callback(10, 100, "正在划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 记录原始样本分布
    original_distribution = {
        'good_count': int((y_train == 0).sum()),
        'bad_count': int((y_train == 1).sum()),
        'total': len(y_train),
        'imbalance_ratio': float((y_train == 1).sum() / (y_train == 0).sum()) if (y_train == 0).sum() > 0 else 0
    }
    
    # 获取模型配置和默认参数（需要在处理样本不平衡之前定义）
    ModelClass = config['class']
    default_params = config['default_params'].copy()
    param_grid = config['param_grid'].copy()
    param_ranges = config.get('param_ranges', {})
    
    # 处理样本不平衡
    if handle_imbalance:
        if progress_callback:
            progress_callback(12, 100, f"正在处理样本不平衡（方法: {imbalance_method}）...")
        
        if imbalance_method == 'class_weight':
            # 计算类别权重
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))
            
            # 为模型设置类别权重
            if model_name == 'LightGBM':
                default_params['class_weight'] = class_weight_dict
            elif model_name == 'XGBoost':
                # XGBoost使用scale_pos_weight
                pos_count = (y_train == 1).sum()
                neg_count = (y_train == 0).sum()
                default_params['scale_pos_weight'] = neg_count / pos_count if pos_count > 0 else 1
            elif model_name == 'Random Forest':
                default_params['class_weight'] = 'balanced'
            elif model_name == 'CatBoost':
                # CatBoost使用class_weights参数
                default_params['class_weights'] = class_weight_dict
            
            processed_distribution = original_distribution.copy()
            
        elif imbalance_method in ['smote', 'adasyn']:
            if not IMBLEARN_AVAILABLE:
                raise ImportError("imbalanced-learn未安装，请运行: pip install imbalanced-learn")
            
            if imbalance_method == 'smote':
                sampler = SMOTE(random_state=42)
            else:  # adasyn
                sampler = ADASYN(random_state=42)
            
            # 保存原始数据类型
            is_dataframe = hasattr(X_train, 'iloc')
            is_series = hasattr(y_train, 'iloc')
            
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            
            # 如果原来是DataFrame/Series，转换回去
            if is_dataframe:
                import pandas as pd
                X_train = pd.DataFrame(X_train, columns=feature_names)
            if is_series:
                import pandas as pd
                y_train = pd.Series(y_train)
            
            processed_distribution = {
                'good_count': int((y_train == 0).sum()),
                'bad_count': int((y_train == 1).sum()),
                'total': len(y_train),
                'imbalance_ratio': float((y_train == 1).sum() / (y_train == 0).sum()) if (y_train == 0).sum() > 0 else 0
            }
        else:
            processed_distribution = original_distribution.copy()
    else:
        processed_distribution = original_distribution.copy()
    
    training_info = {
        'model_type': model_name,
        'use_cv': use_cv,
        'cv_folds': cv_folds if use_cv else None,
        'tune_hyperparams': tune_hyperparams,
        'optimization_method': optimization_method if tune_hyperparams else None,
        'handle_imbalance': handle_imbalance,
        'imbalance_method': imbalance_method if handle_imbalance else None,
        'original_distribution': original_distribution,
        'processed_distribution': processed_distribution,
        'best_params': None,
        'cv_scores': None
    }
    
    if tune_hyperparams and use_cv:
        if optimization_method == 'bayesian':
            # 使用贝叶斯优化（Optuna）
            if progress_callback:
                progress_callback(20, 100, f"正在进行{model_name}贝叶斯优化（{n_trials}次试验，{cv_folds}折交叉验证）...")
            
            best_params, best_cv_score, study = bayesian_optimization(
                X_train, y_train, ModelClass, default_params, param_ranges, 
                cv_folds, n_trials=n_trials, progress_callback=progress_callback
            )
            
            if progress_callback:
                progress_callback(80, 100, "优化完成，正在训练最终模型...")
            
            # 使用最佳参数训练最终模型
            model = ModelClass(**best_params)
            model.fit(X_train, y_train)
            
            training_info['best_params'] = best_params
            training_info['cv_scores'] = {
                'mean_cv_score': best_cv_score,
                'std_cv_score': 0.0,  # Optuna不直接提供标准差
                'all_cv_scores': []
            }
        else:
            # 使用GridSearchCV
            if progress_callback:
                progress_callback(20, 100, f"正在进行{model_name}超参数调优（GridSearchCV，{cv_folds}折交叉验证）...")
            
            # 创建模型实例
            base_estimator = ModelClass(**default_params)
            
            # 使用AUC作为评分标准（修复FutureWarning）
            scorer = make_scorer(roc_auc_score, response_method='predict_proba')
            
            # GridSearchCV with K折交叉验证
            grid_search = GridSearchCV(
                estimator=base_estimator,
                param_grid=param_grid,
                scoring=scorer,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                n_jobs=-1,
                verbose=0  # 减少输出，通过进度条显示
            )
            
            # 训练
            if progress_callback:
                progress_callback(30, 100, f"正在训练{model_name}模型（{cv_folds}折交叉验证）...")
            grid_search.fit(X_train, y_train)
            
            if progress_callback:
                progress_callback(80, 100, "训练完成，正在评估模型...")
            
            # 获取最佳模型和参数
            model = grid_search.best_estimator_
            training_info['best_params'] = grid_search.best_params_
            training_info['cv_scores'] = {
                'mean_cv_score': grid_search.best_score_,
                'std_cv_score': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
                'all_cv_scores': grid_search.cv_results_['mean_test_score'].tolist()
            }
        
    elif tune_hyperparams and not use_cv:
        # 只进行超参数调优，不使用交叉验证（使用train_test_split）
        if progress_callback:
            progress_callback(20, 100, f"正在进行{model_name}超参数调优（不使用交叉验证）...")
        
        # 创建模型实例
        base_estimator = ModelClass(**default_params)
        
        # 使用AUC作为评分标准
        scorer = make_scorer(roc_auc_score, response_method='predict_proba')
        
        # GridSearchCV with 简单的train_test_split
        from sklearn.model_selection import ShuffleSplit
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            scoring=scorer,
            cv=ShuffleSplit(n_splits=3, test_size=0.2, random_state=42),
            n_jobs=-1,
            verbose=0
        )
        
        # 训练
        if progress_callback:
            progress_callback(30, 100, f"正在训练{model_name}模型...")
        grid_search.fit(X_train, y_train)
        
        if progress_callback:
            progress_callback(80, 100, "训练完成，正在评估模型...")
        
        # 获取最佳模型和参数
        model = grid_search.best_estimator_
        training_info['best_params'] = grid_search.best_params_
        training_info['cv_scores'] = {
            'mean_cv_score': grid_search.best_score_,
            'std_cv_score': grid_search.cv_results_['std_test_score'][grid_search.best_index_],
            'all_cv_scores': grid_search.cv_results_['mean_test_score'].tolist()
        }
        
    elif use_cv and not tune_hyperparams:
        # 只使用K折交叉验证，不调优
        if progress_callback:
            progress_callback(20, 100, f"使用{cv_folds}折交叉验证训练{model_name}模型...")
        
        # K折交叉验证
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            if progress_callback:
                progress_callback(20 + int(50 * (fold + 1) / cv_folds), 100, 
                                f"正在训练第 {fold+1}/{cv_folds} 折...")
            
            X_train_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_train_fold = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            y_val_fold = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
            
            fold_model = ModelClass(**default_params)
            fold_model.fit(X_train_fold, y_train_fold)
            
            y_val_pred = fold_model.predict_proba(X_val_fold)[:, 1]
            fold_auc = roc_auc_score(y_val_fold, y_val_pred)
            cv_scores.append(fold_auc)
        
        training_info['cv_scores'] = {
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'all_cv_scores': cv_scores
        }
        
        if progress_callback:
            progress_callback(80, 100, "交叉验证完成，正在训练最终模型...")
        
        # 在全部训练集上训练最终模型
        model = ModelClass(**default_params)
        model.fit(X_train, y_train)
        
    else:
        # 不使用交叉验证，也不调优
        if progress_callback:
            progress_callback(20, 100, f"使用标准方法训练{model_name}模型...")
        
        model = ModelClass(**default_params)
        model.fit(X_train, y_train)
        
        if progress_callback:
            progress_callback(80, 100, "训练完成，正在评估模型...")
    
    # 评估模型
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    
    if progress_callback:
        progress_callback(90, 100, "正在保存模型...")
    
    # 保存模型
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(backend_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'base_model.pkl')
    save_model(model, model_path)
    
    # 保存训练数据用于后续漂移检测
    train_data_path = os.path.join(model_dir, 'base_train_data.pkl')
    save_model({'X_train': X_train, 'y_train': y_train, 'feature_names': feature_names}, 
               train_data_path)
    
    if progress_callback:
        progress_callback(100, 100, "训练完成！")
    
    return {
        'model': model,
        'auc': auc,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'feature_names': feature_names,
        'training_info': training_info
    }
