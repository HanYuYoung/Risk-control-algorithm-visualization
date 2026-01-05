"""
工具函数：数据加载、特征漂移计算、PSI计算等
"""
import pandas as pd
import numpy as np
from scipy import stats
import pickle
import os


def load_prosper_data():
    """
    加载Prosper贷款数据集
    数据来源：Prosper平台真实贷款数据
    Label: LoanStatus (Chargedoff/Defaulted=1坏用户, Completed/Current=0好用户)
    """
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(backend_dir, 'data', 'prosperLoanData.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Prosper数据文件不存在: {data_path}")
    
    # 读取数据（采样以提高速度，实际可用全部数据）
    df = pd.read_csv(data_path, low_memory=False)
    
    # 采样：如果数据量太大，随机采样10万条
    if len(df) > 100000:
        df = df.sample(n=100000, random_state=42).reset_index(drop=True)
    
    return df


def preprocess_data(df, is_drift=False, drift_strength=0.3):
    """
    数据预处理和特征工程 - Prosper数据
    is_drift: 是否模拟特征漂移
    drift_strength: 漂移强度 (0-1)
    """
    df = df.copy()
    
    # 目标变量：LoanStatus -> 转换为0=好用户, 1=坏用户
    # 坏用户：Chargedoff, Defaulted
    # 好用户：Completed, Current
    # 其他状态（Past Due等）过滤掉
    bad_status = ['Chargedoff', 'Defaulted']
    good_status = ['Completed', 'Current']
    
    # 只保留明确的好用户和坏用户
    df = df[df['LoanStatus'].isin(bad_status + good_status)].copy()
    df['target'] = df['LoanStatus'].apply(lambda x: 1 if x in bad_status else 0)
    
    # 选择重要特征（数值特征）
    numeric_features = [
        'BorrowerAPR', 'BorrowerRate', 'ProsperScore', 'Term',
        'LoanOriginalAmount', 'MonthlyLoanPayment', 'DebtToIncomeRatio',
        'CreditScoreRangeLower', 'CreditScoreRangeUpper', 'CurrentCreditLines',
        'OpenCreditLines', 'TotalCreditLinespast7years', 'OpenRevolvingAccounts',
        'InquiriesLast6Months', 'CurrentDelinquencies', 'AmountDelinquent',
        'RevolvingCreditBalance', 'BankcardUtilization', 'TotalTrades',
        'StatedMonthlyIncome', 'EmploymentStatusDuration'
    ]
    
    # 类别特征编码
    categorical_features = [
        'ProsperRating (Alpha)', 'EmploymentStatus', 'IsBorrowerHomeowner',
        'IncomeRange', 'IncomeVerifiable', 'BorrowerState'
    ]
    
    # 处理类别特征
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            df[col] = pd.Categorical(df[col]).codes
    
    # 选择存在的特征
    all_features = numeric_features + categorical_features
    feature_cols = [col for col in all_features if col in df.columns]
    
    # 处理数值特征缺失值
    for col in numeric_features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    # 模拟特征漂移：改变特征分布
    if is_drift:
        np.random.seed(42)
        for col in numeric_features:
            if col in df.columns:
                # 添加漂移：改变均值和方差
                shift = drift_strength * df[col].std()
                df[col] = df[col] + np.random.normal(shift, shift * 0.5, len(df))
                # 保持合理范围
                if df[col].min() < 0:
                    df[col] = np.maximum(df[col], 0)
        
        # 类别特征也进行一些漂移
        for col in categorical_features:
            if col in df.columns and df[col].nunique() > 1:
                # 随机改变一些类别
                mask = np.random.random(len(df)) < drift_strength * 0.3
                df.loc[mask, col] = np.random.choice(df[col].unique(), mask.sum())
    
    # 提取特征和目标
    X = df[feature_cols].fillna(0)
    y = df['target']
    
    return X, y, feature_cols


def calculate_psi(expected, actual, bins=10):
    """
    计算PSI (Population Stability Index) 用于特征漂移检测
    PSI < 0.1: 无显著漂移
    0.1 <= PSI < 0.25: 轻微漂移
    PSI >= 0.25: 显著漂移
    """
    # 合并数据确定分箱边界
    combined = np.concatenate([expected, actual])
    breakpoints = np.linspace(combined.min(), combined.max(), bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    
    # 计算期望分布
    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    expected_pct = expected_counts / len(expected)
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)  # 避免除零
    
    # 计算实际分布
    actual_counts, _ = np.histogram(actual, bins=breakpoints)
    actual_pct = actual_counts / len(actual)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    # 计算PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return psi, expected_pct, actual_pct, breakpoints


def detect_feature_drift(X_train, X_test, feature_names, psi_threshold=0.25):
    """
    检测特征漂移
    返回漂移特征列表和PSI值
    """
    drift_results = {}
    drifted_features = []
    
    for i, feature in enumerate(feature_names):
        expected = X_train.iloc[:, i].values if hasattr(X_train, 'iloc') else X_train[:, i]
        actual = X_test.iloc[:, i].values if hasattr(X_test, 'iloc') else X_test[:, i]
        
        psi, _, _, _ = calculate_psi(expected, actual)
        drift_results[feature] = {
            'psi': psi,
            'drifted': psi >= psi_threshold
        }
        
        if psi >= psi_threshold:
            drifted_features.append(feature)
    
    return drift_results, drifted_features


def save_model(model, filepath):
    """保存模型"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    """加载模型"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

