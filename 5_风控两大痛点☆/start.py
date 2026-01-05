"""
快速启动脚本
"""
import subprocess
import sys
import os

def main():
    # 切换到frontend目录
    frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
    os.chdir(frontend_dir)
    
    # 运行streamlit
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])

if __name__ == '__main__':
    main()



