"""
Core components for TTT methods

This module contains core functionality required by various Test-Time Training approaches,
including configuration management and other foundational components.
"""

# 移除相对导入，使用绝对导入
from core.config import Config

__all__ = [
    'Config'
]