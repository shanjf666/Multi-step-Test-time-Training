"""
Implementation of various TTT methods

This module contains implementations of different Test-Time Training methods including:
- Baseline evaluation
- Self-certainty based selection
- Self-evaluation based selection
- CoE-C (Change-over-Epoch based on Cosine similarity) selection
- Self-consistency based selection
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