"""
Utility functions for TTT methods

This module provides common utility functions used across different TTT methods,
including text processing, answer extraction, step parsing, and confidence calculation helpers.
"""

# 将相对导入改为绝对导入
from utils.common import (
    parse_structured_steps,
    extract_model_answer,
    is_correct_answer,
    generate_with_transformers,
    calculate_step_confidence_with_self_certainty,
    calculate_step_confidence_with_self_eval,
    calculate_step_confidence_with_CoE_C,
    calculate_step_confidence_with_entropy  # 添加这一行
)

from utils.key_step_extractor import (
    summarize_key_steps_openai
)

__all__ = [
    'parse_structured_steps',
    'extract_model_answer',
    'is_correct_answer',
    'generate_with_transformers',
    'calculate_step_confidence_with_self_certainty',
    'calculate_step_confidence_with_self_eval',
    'calculate_step_confidence_with_CoE_C',
    'calculate_step_confidence_with_entropy',  # 添加这一行
    'summarize_key_steps_openai'
]