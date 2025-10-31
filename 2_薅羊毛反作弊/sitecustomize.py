"""
Auto-loaded by Python if present on sys.path.
Disables creation of __pycache__/ and .pyc bytecode files project-wide.
"""
import os
import sys

# Do not write .pyc files
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


