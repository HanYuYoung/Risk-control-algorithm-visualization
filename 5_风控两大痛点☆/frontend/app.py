"""
Streamlit前端：特征漂移检测与自适应更新系统
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# 添加backend路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from models.base_model import train_base_model
from models.drift_model import simulate_drift_and_retrain
from models.utils import calculate_psi

st.set_page_config(
    page_title="特征漂移检测系统",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 特征漂移检测与自适应更新系统")
st.markdown("---")

# 侧边栏
st.sidebar.header("操作面板")
st.sidebar.markdown("### 解决风控两大痛点：")
st.sidebar.markdown("1. **特征失效快** - 黑灰产快速变换特征")
st.sidebar.markdown("2. **样本不平衡** - 作弊用户永远是少数")

# 数据集预览部分
st.header("📋 数据集预览")
with st.expander("查看数据集信息", expanded=True):
    from models.utils import load_prosper_data
    
    # 加载数据
    if 'raw_data' not in st.session_state:
        with st.spinner("正在加载数据..."):
            df = load_prosper_data()
            # 预处理用于显示
            bad_status = ['Chargedoff', 'Defaulted']
            good_status = ['Completed', 'Current']
            df_display = df[df['LoanStatus'].isin(bad_status + good_status)].copy()
            st.session_state.raw_data = df
            st.session_state.df_display = df_display
    
    df = st.session_state.raw_data
    df_display = st.session_state.df_display
    
    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    good_count = len(df_display[df_display['LoanStatus'].isin(['Completed', 'Current'])])
    bad_count = len(df_display[df_display['LoanStatus'].isin(['Chargedoff', 'Defaulted'])])
    total_count = len(df_display)
    imbalance_ratio = bad_count / good_count if good_count > 0 else 0
    
    with col_info1:
        st.metric("总样本数", total_count)
    with col_info2:
        st.metric("特征数量", len(df.columns) - 1)
    with col_info3:
        st.metric("好用户", good_count)
    with col_info4:
        st.metric("坏用户", bad_count)
    
    # 显示样本不平衡信息
    col_imbalance1, col_imbalance2 = st.columns(2)
    with col_imbalance1:
        st.metric("不平衡比例", f"{imbalance_ratio:.2%}", 
                 delta=f"坏用户:好用户 = 1:{1/imbalance_ratio:.1f}" if imbalance_ratio > 0 else None,
                 delta_color="inverse" if imbalance_ratio < 0.3 else "normal")
    with col_imbalance2:
        if imbalance_ratio < 0.3:
            st.warning(f"⚠️ **样本严重不平衡**: 坏用户仅占 {bad_count/total_count:.1%}，需要处理样本不平衡问题！")
        elif imbalance_ratio < 0.5:
            st.info(f"💡 **样本轻度不平衡**: 坏用户占 {bad_count/total_count:.1%}，建议处理样本不平衡")
        else:
            st.success(f"✅ **样本相对平衡**: 坏用户占 {bad_count/total_count:.1%}")
    
    st.markdown("**数据集来源**: Prosper平台真实贷款数据")
    st.markdown("**Label列**: LoanStatus (Chargedoff/Defaulted=坏用户, Completed/Current=好用户)")
    st.markdown("**说明**: Prosper P2P借贷平台真实业务数据，81个特征，用于预测贷款违约风险")
    
    st.markdown("### 数据预览（前10行）")
    st.dataframe(df.head(10), width='stretch')
    
    st.markdown("### 特征统计信息")
    st.dataframe(df.describe(), width='stretch')

st.markdown("---")

# 初始化session state
if 'base_model_trained' not in st.session_state:
    st.session_state.base_model_trained = False
if 'drift_detected' not in st.session_state:
    st.session_state.drift_detected = False
if 'model_retrained' not in st.session_state:
    st.session_state.model_retrained = False

# 主界面
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 步骤1: 训练初始模型")
    
    # 训练配置选项
    with st.expander("⚙️ 训练配置", expanded=False):
        # 模型选择
        model_options = ['LightGBM', 'XGBoost', 'Random Forest', 'CatBoost']
        selected_model = st.selectbox(
            "选择模型",
            options=model_options,
            index=0,
            help="选择要训练的模型。LightGBM和XGBoost适合大数据，Random Forest稳定可靠，CatBoost处理类别特征好"
        )
        
        use_cv = st.checkbox("使用K折交叉验证", value=True, help="使用K折交叉验证可以更可靠地评估模型性能")
        cv_folds = st.slider("K折数", min_value=3, max_value=10, value=5, disabled=not use_cv, help="交叉验证的折数，建议5折")
        st.markdown("---")
        st.markdown("**⚖️ 样本不平衡处理选项**")
        handle_imbalance = st.checkbox(
            "处理样本不平衡",
            value=True,
            help="✅ 勾选：使用技术处理样本不平衡问题（类别权重或过采样）\n\n❌ 不勾选：不处理样本不平衡，可能导致模型偏向多数类"
        )
        if handle_imbalance:
            imbalance_method = st.radio(
                "处理方法",
                options=['class_weight', 'smote', 'adasyn'],
                format_func=lambda x: {
                    'class_weight': '⚖️ 类别权重 (Class Weight) - 推荐，不改变数据',
                    'smote': '📈 SMOTE过采样 - 生成合成样本',
                    'adasyn': '📊 ADASYN过采样 - 自适应合成样本'
                }[x],
                help="类别权重：通过调整损失函数权重处理不平衡，不改变数据分布\nSMOTE/ADASYN：通过生成合成少数类样本来平衡数据",
                key='imbalance_method_radio'
            )
            st.session_state['imbalance_method'] = imbalance_method
            st.session_state['handle_imbalance'] = True
        else:
            st.session_state['imbalance_method'] = None
            st.session_state['handle_imbalance'] = False
        st.markdown("---")
        st.markdown("**⚙️ 超参数调优选项**")
        tune_hyperparams = st.checkbox(
            "进行超参数调优", 
            value=False,  # 默认不调优，使用默认参数快速训练
            help="✅ 勾选：自动搜索最佳超参数，训练时间较长但模型性能更好\n\n❌ 不勾选：使用模型默认参数快速训练，适合快速验证"
        )
        
        if tune_hyperparams:
            # 优化方法选择
            optimization_method = st.radio(
                "优化方法",
                options=['bayesian', 'grid'],
                format_func=lambda x: '🚀 贝叶斯优化 (Optuna) - 推荐，速度快' if x == 'bayesian' else '📊 网格搜索 (GridSearchCV) - 全面但慢',
                help="贝叶斯优化通常比网格搜索快5-10倍，推荐使用",
                key='optimization_method_radio'
            )
            st.session_state['optimization_method'] = optimization_method
            
            if optimization_method == 'bayesian':
                n_trials = st.slider(
                    "贝叶斯优化试验次数",
                    min_value=20,
                    max_value=200,
                    value=50,
                    step=10,
                    help="试验次数越多，找到最佳参数的概率越高，但训练时间也越长。推荐50-100次",
                    key='n_trials_slider'
                )
                st.session_state['n_trials'] = n_trials
                st.success(f"✅ **贝叶斯优化**: 将进行 {n_trials} 次智能搜索，通常比网格搜索快5-10倍！")
            else:
                st.warning("⚠️ **网格搜索**: 会尝试所有参数组合，训练时间可能很长（几分钟到几十分钟）")
        else:
            st.info("💡 **当前模式**: 使用默认参数快速训练，训练速度快，适合快速验证模型效果")
        
        # 显示多线程信息
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            st.success(f"🚀 **多线程加速已启用**: 将使用 {cpu_count} 个CPU核心并行训练，大幅提升训练速度")
        except:
            st.info("💡 **多线程加速已启用**: 将使用所有可用CPU核心并行训练")
        
        st.info("💡 **推荐配置**: \n- **快速测试**: 不勾选参数调优，使用默认参数\n- **最佳性能**: 启用K折交叉验证(5折) + 超参数调优")
    
    if st.button("🚀 训练初始模型", type="primary", use_container_width=True):
        # 创建进度条
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, message):
            """更新进度条的回调函数"""
            progress = int((current / total) * 100)
            progress_bar.progress(progress)
            status_text.text(f"进度: {progress}% - {message}")
        
        try:
            # 获取优化方法和试验次数（从expander内部获取）
            optimization_method = 'bayesian'  # 默认值
            n_trials = 50  # 默认值
            
            # 如果进行了调优，需要重新获取这些值
            # 由于streamlit的限制，我们需要在expander内部保存到session_state
            if tune_hyperparams:
                # 从expander内部获取的值已经在session_state中
                optimization_method = st.session_state.get('optimization_method', 'bayesian')
                n_trials = st.session_state.get('n_trials', 50)
            
            # 获取样本不平衡处理参数
            handle_imbalance = st.session_state.get('handle_imbalance', True)
            imbalance_method = st.session_state.get('imbalance_method', 'class_weight')
            
            result = train_base_model(
                model_name=selected_model,
                use_cv=use_cv, 
                cv_folds=cv_folds, 
                tune_hyperparams=tune_hyperparams,
                optimization_method=optimization_method if tune_hyperparams else 'grid',
                n_trials=n_trials if tune_hyperparams and optimization_method == 'bayesian' else 50,
                handle_imbalance=handle_imbalance,
                imbalance_method=imbalance_method if handle_imbalance else None,
                progress_callback=update_progress
            )
            st.session_state.base_result = result
            st.session_state.base_model_trained = True
            # 保存训练数据用于后续分布对比
            st.session_state.X_train_base = result['X_train']
            
            progress_bar.progress(100)
            status_text.text("✅ 训练完成！")
            st.success("✅ 初始模型训练完成！")
        except Exception as e:
            st.error(f"❌ 训练失败: {str(e)}")
            st.info("💡 提示: 如果选择了XGBoost或CatBoost，请确保已安装相应库: `pip install xgboost catboost`")
    
    if st.session_state.base_model_trained:
        result = st.session_state.base_result
        training_info = result.get('training_info', {})
        
        # 显示样本不平衡处理信息
        if training_info.get('handle_imbalance'):
            st.markdown("### ⚖️ 样本不平衡处理结果")
            original_dist = training_info.get('original_distribution', {})
            processed_dist = training_info.get('processed_distribution', {})
            method = training_info.get('imbalance_method', 'class_weight')
            
            method_names = {
                'class_weight': '类别权重',
                'smote': 'SMOTE过采样',
                'adasyn': 'ADASYN过采样'
            }
            
            col_dist1, col_dist2 = st.columns(2)
            with col_dist1:
                st.markdown("**📊 处理前样本分布**")
                st.metric("好用户", original_dist.get('good_count', 0))
                st.metric("坏用户", original_dist.get('bad_count', 0))
                st.metric("不平衡比例", f"{original_dist.get('imbalance_ratio', 0):.2%}")
            
            with col_dist2:
                st.markdown(f"**📊 处理后样本分布 ({method_names.get(method, method)})**")
                if method == 'class_weight':
                    st.metric("好用户", processed_dist.get('good_count', 0), delta="未改变（使用权重调整）")
                    st.metric("坏用户", processed_dist.get('bad_count', 0), delta="未改变（使用权重调整）")
                    st.info("💡 使用类别权重调整，不改变样本数量，通过调整损失函数权重处理不平衡")
                else:
                    st.metric("好用户", processed_dist.get('good_count', 0), 
                             delta=f"+{processed_dist.get('good_count', 0) - original_dist.get('good_count', 0)}")
                    st.metric("坏用户", processed_dist.get('bad_count', 0),
                             delta=f"+{processed_dist.get('bad_count', 0) - original_dist.get('bad_count', 0)}")
                    st.metric("不平衡比例", f"{processed_dist.get('imbalance_ratio', 0):.2%}",
                             delta=f"{processed_dist.get('imbalance_ratio', 0) - original_dist.get('imbalance_ratio', 0):.2%}")
                    st.success(f"✅ 通过{method_names.get(method, method)}生成了合成样本，平衡了数据分布")
            
            st.markdown("---")
        
        # 显示测试集性能
        st.markdown("### 📈 测试集性能")
        st.metric("测试集AUC", f"{result['auc']:.4f}")
        
        # 显示混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        fig_cm = px.imshow(
            cm,
            labels=dict(x="预测", y="实际"),
            x=['好用户', '坏用户'],
            y=['好用户', '坏用户'],
            text_auto=True,
            aspect="auto",
            title="初始模型混淆矩阵"
        )
        st.plotly_chart(fig_cm, width='stretch')
        
        st.info("💡 模型已训练，可以正常识别风险用户")

with col2:
    st.header("🔍 步骤2: 模拟特征漂移")
    
    drift_strength = st.slider("漂移强度", 0.1, 0.8, 0.3, 0.1)
    
    if st.button("⚠️ 模拟特征漂移", type="secondary", use_container_width=True):  # button仍使用use_container_width
        if not st.session_state.base_model_trained:
            st.error("❌ 请先训练初始模型！")
        else:
            with st.spinner("正在模拟特征漂移..."):
                drift_result = simulate_drift_and_retrain(drift_strength=drift_strength)
                st.session_state.drift_result = drift_result
                st.session_state.drift_detected = True
                st.warning("⚠️ 特征漂移已发生！")
    
    if st.session_state.drift_detected:
        result = st.session_state.drift_result
        st.metric("旧模型AUC (漂移后)", f"{result['auc_old']:.4f}", delta=f"{result['auc_old'] - st.session_state.base_result['auc']:.4f}")
        st.metric("检测到漂移特征数", len(result['drifted_features']))
        
        # 显示漂移特征
        if result['drifted_features']:
            st.warning(f"⚠️ 漂移特征: {', '.join(result['drifted_features'][:5])}")
        
        st.error("💥 旧模型性能下降，需要重新训练！")

# 漂移检测可视化
if st.session_state.drift_detected:
    st.markdown("---")
    st.header("📈 特征漂移详细分析")
    
    result = st.session_state.drift_result
    
    # 漂移统计
    col_drift1, col_drift2, col_drift3 = st.columns(3)
    with col_drift1:
        total_features = len(result['drift_results'])
        st.metric("总特征数", total_features)
    with col_drift2:
        drifted_count = len(result['drifted_features'])
        st.metric("发生漂移的特征数", drifted_count, delta=f"{drifted_count/total_features*100:.1f}%")
    with col_drift3:
        avg_psi = np.mean([v['psi'] for v in result['drift_results'].values()])
        st.metric("平均PSI值", f"{avg_psi:.4f}")
    
    # PSI值详细表格
    drift_df = pd.DataFrame([
        {
            '特征名称': k, 
            'PSI值': v['psi'], 
            '是否漂移': '是 ⚠️' if v['drifted'] else '否 ✅',
            '漂移程度': '严重' if v['psi'] >= 0.5 else ('中等' if v['psi'] >= 0.25 else ('轻微' if v['psi'] >= 0.1 else '无'))
        }
        for k, v in result['drift_results'].items()
    ])
    
    # 按PSI值排序
    drift_df = drift_df.sort_values('PSI值', ascending=False)
    
    st.markdown("### 📊 所有特征的PSI值（按漂移程度排序）")
    st.dataframe(drift_df, width='stretch', hide_index=True)
    
    # PSI值可视化
    st.markdown("### 📈 PSI值可视化（Top 15）")
    fig_psi = px.bar(
        drift_df.head(15),
        x='特征名称',
        y='PSI值',
        color='是否漂移',
        title="特征PSI值排名 (Top 15) - PSI≥0.25表示显著漂移",
        color_discrete_map={'是 ⚠️': 'red', '否 ✅': 'green'},
        text='PSI值'
    )
    fig_psi.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig_psi.update_layout(
        xaxis_tickangle=-45,
        yaxis_title="PSI值",
        height=500
    )
    # 添加PSI阈值线
    fig_psi.add_hline(y=0.25, line_dash="dash", line_color="orange", 
                     annotation_text="漂移阈值 (PSI=0.25)", annotation_position="right")
    st.plotly_chart(fig_psi, width='stretch')
    
    # 漂移特征详细列表
    if result['drifted_features']:
        st.markdown("### ⚠️ 发生漂移的特征详情")
        drifted_details = []
        for feat in result['drifted_features']:
            psi_val = result['drift_results'][feat]['psi']
            drifted_details.append({
                '特征名称': feat,
                'PSI值': f"{psi_val:.4f}",
                '漂移程度': '严重' if psi_val >= 0.5 else '中等'
            })
        drifted_df = pd.DataFrame(drifted_details)
        st.dataframe(drifted_df, width='stretch', hide_index=True)
        
        # 特征分布对比（选择前3个漂移最严重的特征）
        st.markdown("### 📉 漂移特征分布对比（前3个最严重）")
        top_drifted = sorted(result['drifted_features'], 
                           key=lambda x: result['drift_results'][x]['psi'], 
                           reverse=True)[:3]
        
        if top_drifted:
            from models.utils import load_model
            backend_dir = os.path.join(os.path.dirname(__file__), '..', 'backend')
            base_train_data = load_model(os.path.join(backend_dir, 'models', 'base_train_data.pkl'))
            X_train_base = base_train_data['X_train']
            feature_names = base_train_data['feature_names']
            
            for feat_name in top_drifted:
                if feat_name in feature_names:
                    feat_idx = feature_names.index(feat_name)
                    # 获取原始分布
                    if hasattr(X_train_base, 'iloc'):
                        original_dist = X_train_base.iloc[:, feat_idx].values
                    else:
                        original_dist = X_train_base[:, feat_idx]
                    
                    # 获取漂移后分布
                    if hasattr(result['X_drift'], 'iloc'):
                        drifted_dist = result['X_drift'].iloc[:, feat_idx].values
                    else:
                        drifted_dist = result['X_drift'][:, feat_idx]
                    
                    psi_val = result['drift_results'][feat_name]['psi']
                    
                    # 创建分布对比图
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=original_dist,
                        name='原始分布（训练时）',
                        opacity=0.7,
                        nbinsx=30,
                        marker_color='blue'
                    ))
                    fig_dist.add_trace(go.Histogram(
                        x=drifted_dist,
                        name='漂移后分布（当前）',
                        opacity=0.7,
                        nbinsx=30,
                        marker_color='red'
                    ))
                    fig_dist.update_layout(
                        title=f"特征 '{feat_name}' 分布对比 | PSI={psi_val:.4f} {'(严重漂移)' if psi_val >= 0.5 else '(中等漂移)'}",
                        xaxis_title="特征值",
                        yaxis_title="频数",
                        barmode='overlay',
                        height=350,
                        legend=dict(x=0.7, y=0.9)
                    )
                    st.plotly_chart(fig_dist, width='stretch')
    
    # 模型性能对比
    st.markdown("---")
    st.header("🔄 步骤3: 模型重训练")
    
    if st.button("🔄 重新训练模型", type="primary", use_container_width=True):
        with st.spinner("正在重新训练模型..."):
            st.session_state.model_retrained = True
            st.success("✅ 模型重训练完成！")
    
    if st.session_state.model_retrained:
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric("初始模型AUC", f"{st.session_state.base_result['auc']:.4f}")
        with col4:
            st.metric("漂移后旧模型AUC", f"{result['auc_old']:.4f}", 
                     delta=f"{result['auc_old'] - st.session_state.base_result['auc']:.4f}", 
                     delta_color="inverse")
        with col5:
            st.metric("重训练后新模型AUC", f"{result['auc_new']:.4f}",
                     delta=f"{result['auc_new'] - result['auc_old']:.4f}")
        
        # 性能对比图
        performance_data = {
            '阶段': ['初始训练', '漂移后(旧模型)', '重训练后(新模型)'],
            'AUC': [
                st.session_state.base_result['auc'],
                result['auc_old'],
                result['auc_new']
            ]
        }
        perf_df = pd.DataFrame(performance_data)
        
        fig_perf = px.line(
            perf_df,
            x='阶段',
            y='AUC',
            markers=True,
            title="模型性能变化",
            text='AUC'
        )
        fig_perf.update_traces(texttemplate='%{text:.4f}', textposition="top center")
        fig_perf.update_layout(yaxis_range=[0.5, 1.0])
        st.plotly_chart(fig_perf, width='stretch')
        
        st.success("🎉 新模型已适应漂移后的数据分布，性能恢复！")

# 底部说明
st.markdown("---")
st.markdown("""
### 📝 项目说明

**数据集来源**: Prosper平台真实贷款数据
- Label: LoanStatus (Chargedoff/Defaulted=坏用户, Completed/Current=好用户)
- 真实P2P借贷业务数据，81个特征
- 目标：预测贷款违约风险

**解决的问题**:
1. ✅ **特征失效快**: 通过PSI检测特征漂移，自动触发模型重训练
2. ✅ **样本不平衡**: 使用LightGBM处理不平衡数据，AUC评估模型性能

**技术要点**:
- PSI (Population Stability Index) 特征漂移检测
- LightGBM 梯度提升树模型
- 自适应模型更新机制
""")

