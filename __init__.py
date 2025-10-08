"""
TTT (Test-Time Training) Methods Package

A modular implementation of various Test-Time Training methods for improving
reasoning capabilities of large language models.
"""

# 核心组件
from .core.config import Config

# 主要方法
from .methods.baseline import baseline_evaluation
from .methods.self_certainty import Self_Certainty_Selection
from .methods.self_eval import Self_Eval_Selection
from .methods.coe_c import CoE_C_Selection
from .methods.self_consistency import Self_Consistency_Selection

# 主入口点
from .main import main

__all__ = [
    'Config',
    'baseline_evaluation',
    'Self_Certainty_Selection',
    'Self_Eval_Selection',
    'CoE_C_Selection',
    'Self_Consistency_Selection',
    'main'
]
