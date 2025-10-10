"""
Key step extraction utilities using OpenAI-compatible APIs
"""

import re
import math

KEY_STEP_PROMPT_TEMPLATE = """You are a reasoning process analyst.

Your task is to extract the *key reasoning steps* (v1, v2, ...) from the given reasoning trace.
Each key step should represent an essential logical or mathematical operation that directly contributes to the final answer.
And do not judge the correctness of the reasoning.
Follow these principles (based on StepRAR methodology):
1. Identify only **essential reasoning variables or equations** that move the solution forward.
2. Remove redundant or narrative sentences.
3. Retain minimal but complete steps (e.g., "16 - 3 = 13", "13 - 4 = 9", "9 * 2 = 18").
4. Unify equivalent math expressions ("6/3=2", "6 divided by 3 equals 2") into a **canonical format**.
5. Provide the output in the format like:
  key_steps: ##Step 1: ..., ##Step 2: ..., ...,final_answer: ...,

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