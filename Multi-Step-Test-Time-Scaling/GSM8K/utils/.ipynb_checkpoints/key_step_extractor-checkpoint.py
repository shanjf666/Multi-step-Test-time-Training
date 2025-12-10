"""
Key step extraction utilities using OpenAI-compatible APIs
"""

import re
import math

KEY_STEP_PROMPT_TEMPLATE = """You are a reasoning step extraction model for mathematical problems.

Your goal: extract only symbolic or numeric computation steps that directly lead to the answer.

Rules:
1. Remove all narrative or explanatory phrases (e.g., "so", "therefore", "she has").
2. Keep only equations, arithmetic operations, or proportional relationships.
3. For each computation, list equivalent forms: symbolic (e.g., 3×5), decimal (15.0), and verbal (e.g., "3 times 5") and latex (e.g., $3 \\times 5$).
4. Merge trivial transformations (e.g., “= 15” and “equals 15”) into one step.
5. Omit the final answer step.
6. Do not correct any mistakes in the reasoning—extract what is actually used, even if flawed
7. Output in the following JSON format:
{{
  "key_steps": [
    ["16 - 3 = 13"],
    ["13 - 4 = 9"],
    ["9 × 2 = 18", "9 * 2","9 times 2"]
  ]
}}
Now extract the key reasoning steps from the reasoning below:

{reasoning}
"""


def _clean_for_summary(text: str) -> str:
    # 清理模型输出中的控制符与模板残留
    cleaned = re.sub(r'<\|eot_id\|>', '', text)
    cleaned = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<\|.*?\|>', '', cleaned)  # 清除 <|...|>
    cleaned = cleaned.replace("<|end_of_text|>", "")
    cleaned = cleaned.replace("<|end_of_sentence|>", "")
    cleaned = cleaned.replace("</s>", "")
    cleaned = re.sub(r'<｜end▁of▁sentence｜>', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def summarize_key_steps_openai(client: "OpenAI",
                               model: str,
                               reasoning_text: str,
                               temperature: float = 0.2) -> str:
    """
    使用 OpenAI 兼容接口抽取关键步骤（不做本地启发式回退）。
    """
    prompt = KEY_STEP_PROMPT_TEMPLATE.format(reasoning=_clean_for_summary(reasoning_text))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert reasoning summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    content = (resp.choices[0].message.content or "").strip() if resp and resp.choices else ""
    return content