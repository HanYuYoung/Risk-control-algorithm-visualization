"""
Flask API服务器 (moved from api/server.py)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from datetime import datetime
from models.risk_model import RiskModel
from config import MODEL_CONFIG, RISK_THRESHOLD_LOW, RISK_THRESHOLD_HIGH, API_PORT, DATA_DIR, LOGS_DIR
import os
import json
import threading

app = Flask(__name__)
CORS(app)

# 全局风控模型
risk_model = RiskModel()

# 请求历史（用于统计）
request_history = []
history_lock = threading.Lock()

# 当前运行的攻击模拟器实例（用于停止）
current_attacker = None
attacker_lock = threading.Lock()
current_log_file = None

@app.route('/api/register', methods=['POST'])
def register():
    """注册接口 - 接收注册请求并返回风控结果"""
    try:
        data = request.get_json()
        
        # 添加时间戳
        if 'timestamp' not in data:
            data['timestamp'] = time.time()
        
        # 风控检测
        result = risk_model.predict(data)
        
        # 记录请求（包含真实标签，用于上帝视角对比）
        # 从请求数据中获取真实标签（由模拟器设置）
        true_label = data.get('_true_label', 'normal')  # 如果模拟器设置了_true_label，使用它
        if not true_label or true_label == 'normal':
            # 如果没有标签，通过行为特征推断（向后兼容）
            behavior = data.get('behavior', {})
            path_entropy = behavior.get('path_entropy', 1.0)
            stay_time = behavior.get('page_stay_time', 10)
            true_label = 'attack' if (path_entropy < 0.5 and stay_time < 5) else 'normal'
        
        # 移除内部标记字段
        request_data_clean = {k: v for k, v in data.items() if not k.startswith('_')}
        
        request_record = {
            'timestamp': datetime.now().isoformat(),
            'request': request_data_clean,
            'result': result,
            'true_label': true_label,  # 真实标签（上帝视角）
        }
        
        with history_lock:
            request_history.append(request_record)
            # 只保留最近1000条记录
            if len(request_history) > 1000:
                request_history.pop(0)
        # 追加写入当前日志文件（JSONL）
        try:
            if current_log_file:
                with open(current_log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(request_record, ensure_ascii=False) + '\n')
        except Exception:
            pass
        
        # 返回结果
        return jsonify({
            'success': True,
            'risk_score': result['risk_score'],
            'risk_level': result['risk_level'],
            'risk_type': result['risk_type'],
            'action': get_action(result['risk_level']),
            'details': result.get('details', {}),
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取统计数据"""
    with history_lock:
        recent_requests = request_history[-100:] if len(request_history) > 100 else request_history
        
        stats = {
            'total_requests': len(request_history),
            'recent_requests': len(recent_requests),
            'risk_distribution': {
                'pass': 0,
                'review': 0,
                'reject': 0,
            },
            'request_type_distribution': {
                'normal': 0,
                'attack': 0,
            },
            'risk_type_distribution': {},
            'accuracy_stats': {
                'total': 0,
                'correct': 0,
                'accuracy': 0.0,
                'intercept_rate': 0.0,  # 拦截率（攻击中被拦截的比例）
                'false_positive_rate': 0.0,  # 误报率（正常用户被拦截的比例）
                'false_negative_rate': 0.0,  # 漏检率（攻击未被拦截的比例）
                'true_positive': 0,  # 攻击且被拦截
                'true_negative': 0,  # 正常且通过
                'false_positive': 0,  # 正常但被拦截
                'false_negative': 0,  # 攻击但未拦截
            },
            'recent_results': []
        }
        
        for record in recent_requests:
            risk_level = record['result']['risk_level']
            risk_type = record['result']['risk_type']
            true_label = record.get('true_label', 'normal')
            
            stats['risk_distribution'][risk_level] = stats['risk_distribution'].get(risk_level, 0) + 1
            stats['risk_type_distribution'][risk_type] = stats['risk_type_distribution'].get(risk_type, 0) + 1
            
            # 统计请求类型（真实标签）
            if true_label == 'attack':
                stats['request_type_distribution']['attack'] += 1
            else:
                stats['request_type_distribution']['normal'] += 1
            
            # 计算准确率（上帝视角）
            is_attack = (true_label == 'attack')
            is_blocked = (risk_level == 'reject')
            is_passed = (risk_level == 'pass')
            
            stats['accuracy_stats']['total'] += 1
            
            if is_attack and is_blocked:
                stats['accuracy_stats']['true_positive'] += 1
                stats['accuracy_stats']['correct'] += 1
            elif not is_attack and is_passed:
                stats['accuracy_stats']['true_negative'] += 1
                stats['accuracy_stats']['correct'] += 1
            elif not is_attack and is_blocked:
                stats['accuracy_stats']['false_positive'] += 1
            elif is_attack and (not is_blocked):
                stats['accuracy_stats']['false_negative'] += 1
        
        # 计算准确率指标
        acc_stats = stats['accuracy_stats']
        if acc_stats['total'] > 0:
            acc_stats['accuracy'] = acc_stats['correct'] / acc_stats['total']
            
            attack_count = stats['request_type_distribution']['attack']
            normal_count = stats['request_type_distribution']['normal']
            
            if attack_count > 0:
                acc_stats['intercept_rate'] = acc_stats['true_positive'] / attack_count  # 拦截率
                acc_stats['false_negative_rate'] = acc_stats['false_negative'] / attack_count  # 漏检率
            
            if normal_count > 0:
                acc_stats['false_positive_rate'] = acc_stats['false_positive'] / normal_count  # 误报率
        
        # 最近10条结果
        stats['recent_results'] = [
            {
                'timestamp': r['timestamp'],
                'risk_level': r['result']['risk_level'],
                'risk_score': r['result']['risk_score'],
                'risk_type': r['result']['risk_type'],
            }
            for r in recent_requests[-10:]
        ]
        
        return jsonify(stats), 200

@app.route('/api/history', methods=['GET'])
def get_history():
    """获取请求历史"""
    limit = request.args.get('limit', 50, type=int)
    
    with history_lock:
        history = request_history[-limit:] if len(request_history) > limit else request_history
        
        return jsonify({
            'total': len(request_history),
            'returned': len(history),
            'history': history
        }), 200

@app.route('/api/stop_attack', methods=['POST'])
def stop_attack():
    """停止模拟"""
    try:
        with attacker_lock:
            global current_attacker
            if current_attacker and current_attacker.running:
                current_attacker.stop_attack()
                current_attacker = None
                # 写入停止事件并清空当前日志
                try:
                    if current_log_file:
                        with open(current_log_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps({'event': 'stop_attack', 'timestamp': datetime.now().isoformat()}, ensure_ascii=False) + '\n')
                except Exception:
                    pass
                finally:
                    globals()['current_log_file'] = None
                return jsonify({
                    'success': True,
                    'message': '模拟已停止'
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'message': '没有正在运行的模拟'
                }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'model_loaded': risk_model.is_trained,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/start_attack', methods=['POST'])
def start_attack():
    """启动模拟（从Web UI触发，包含正常用户和羊毛党）"""
    try:
        data = request.get_json() or {}
        thread_count = int(data.get('thread_count', 10))
        requests_per_thread = int(data.get('requests_per_thread', 20))
        attack_ratio = float(data.get('attack_ratio', 0.3))  # 默认30%为攻击请求
        
        print(f"\n[启动模拟] 线程数: {thread_count}, 每线程请求数: {requests_per_thread}, 攻击比例: {attack_ratio}")
        
        # 在新线程中启动模拟
        from simulator.attacker import Attacker
        
        attacker = Attacker(
            api_url=f'http://localhost:{API_PORT}/api/register',
            thread_count=thread_count,
            requests_per_thread=requests_per_thread,
            attack_ratio=attack_ratio
        )
        
        # 保存attacker实例以便停止
        with attacker_lock:
            global current_attacker, current_log_file
            current_attacker = attacker
            # 生成新的日志文件
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'attack_{ts}.jsonl'
            log_path = os.path.join(LOGS_DIR, log_filename)
            try:
                with open(log_path, 'w', encoding='utf-8') as f:
                    header = {
                        'event': 'start_attack',
                        'timestamp': datetime.now().isoformat(),
                        'thread_count': thread_count,
                        'requests_per_thread': requests_per_thread,
                        'attack_ratio': attack_ratio
                    }
                    f.write(json.dumps(header, ensure_ascii=False) + '\n')
                current_log_file = log_path
            except Exception:
                current_log_file = None
        
        attack_thread = threading.Thread(target=attacker.start_attack)
        attack_thread.daemon = True
        attack_thread.start()
        
        total_requests = thread_count * requests_per_thread
        attack_count = int(total_requests * attack_ratio)
        normal_count = total_requests - attack_count
        
        return jsonify({
            'success': True,
            'message': f'模拟已启动: {thread_count}线程 x {requests_per_thread}请求 = {total_requests}总请求',
            'thread_count': thread_count,
            'requests_per_thread': requests_per_thread,
            'attack_ratio': attack_ratio,
            'total_requests': total_requests,
            'expected_normal': normal_count,
            'expected_attack': attack_count,
            'log_file': current_log_file
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def get_action(risk_level):
    """根据风险等级返回处理动作"""
    if risk_level == 'pass':
        return '正常发券'
    elif risk_level == 'review':
        return '加验证码，限流'
    else:
        return '拦截，加入黑名单'

def init_risk_model():
    """初始化风控模型"""
    print("正在加载风控模型...")
    # 兼容老路径：如果新路径不存在而旧路径存在，则优先旧路径
    new_xgb = MODEL_CONFIG['xgb_model_path']
    new_iso = MODEL_CONFIG['isolation_forest_path']
    old_models_dir = os.path.join(DATA_DIR, 'models')
    old_xgb = os.path.join(old_models_dir, 'xgb_model.pkl')
    old_iso = os.path.join(old_models_dir, 'isolation_forest.pkl')

    xgb_path = new_xgb if os.path.exists(new_xgb) or not os.path.exists(old_xgb) else old_xgb
    iso_path = new_iso if os.path.exists(new_iso) or not os.path.exists(old_iso) else old_iso

    risk_model.load_models(
        xgb_path=xgb_path,
        isolation_forest_path=iso_path
    )
    
    if risk_model.is_trained:
        print("[OK] 风控模型加载成功")
    else:
        print("[WARN] 风控模型未训练，将使用基于规则的检测")

if __name__ == '__main__':
    init_risk_model()
    print(f"\n启动API服务器...")
    print(f"访问地址: http://localhost:5000")
    print(f"API文档:")
    print(f"  POST /api/register - 注册接口")
    print(f"  GET  /api/stats    - 统计数据")
    print(f"  GET  /api/history  - 请求历史")
    print(f"  GET  /api/health   - 健康检查")
    print()
    app.run(host='0.0.0.0', port=5000, debug=False)


