"""
公共工具函数
"""
import sys
import os

def setup_encoding():
    """设置Windows环境下的编码"""
    if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass

def setup_path():
    """设置项目路径"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root

