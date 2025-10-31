"""
主应用启动文件
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import threading
from flask import send_file
from web.server import app, init_risk_model
from config import API_HOST, API_PORT
import webbrowser

# 注册首页路由
@app.route('/')
def index():
    """首页"""
    # 获取项目根目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base_dir, 'web', 'index.html')
    
    if os.path.exists(index_path):
        return send_file(index_path, mimetype='text/html')
    else:
        return f"<h1>404 - 页面未找到</h1><p>文件路径: {index_path}</p><p>当前目录: {base_dir}</p>", 404

def open_browser():
    """延迟打开浏览器"""
    time.sleep(2)
    url = f'http://localhost:{API_PORT}'
    webbrowser.open(url)
    print(f"[OK] 浏览器已打开: {url}")

if __name__ == '__main__':
    print("=" * 60)
    print("羊毛党反作弊系统")
    print("=" * 60)
    
    # 初始化模型
    init_risk_model()
    
    # 启动浏览器（在后台线程）
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    print(f"\n启动Web服务器...")
    print(f"访问地址: http://localhost:{API_PORT}")
    print(f"按 Ctrl+C 停止服务\n")
    
    # 启动Flask应用
    app.run(host=API_HOST, port=API_PORT, debug=False, use_reloader=False)

