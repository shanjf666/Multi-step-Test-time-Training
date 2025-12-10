# import json
# from collections import Counter
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 读取 JSONL 文件
# dataset = []
# with open("math500_eval_qwen1.5b_instruct.jsonl", "r") as f:
#     for line in f:
#         dataset.append(json.loads(line.strip()))

# # 统计每条回答的步骤数量和正确性
# step_counts = []
# correct_flags = []

# for item in dataset:
#     answer = item["answers"][0]  # 取第一个答案
#     num_steps = len(answer["steps"])
#     step_counts.append(num_steps)
#     correct_flags.append(answer["correct"])

# # 统计不同步骤数下的正确和错误数量
# count_correct = Counter([step_counts[i] for i in range(len(dataset)) if correct_flags[i]])
# count_incorrect = Counter([step_counts[i] for i in range(len(dataset)) if not correct_flags[i]])

# # 准备数据绘图
# steps = sorted(set(step_counts))
# correct_values = [count_correct.get(s, 0) for s in steps]
# incorrect_values = [count_incorrect.get(s, 0) for s in steps]

# # 可视化
# plt.figure(figsize=(8,5))
# plt.bar([i-0.2 for i in range(len(steps))], correct_values, width=0.4, label="Correct", color='green')
# plt.bar([i+0.2 for i in range(len(steps))], incorrect_values, width=0.4, label="Incorrect", color='red')
# plt.xticks(range(len(steps)), steps)
# plt.xlabel("Number of Steps")
# plt.ylabel("Number of Answers")
# plt.title("Distribution of Number of Steps vs Correctness")
# plt.legend()

# # 保存图片到当前文件夹
# plt.savefig("step_count_vs_correctness.png", dpi=300)
# plt.show()

# print("图已保存为 step_count_vs_correctness.png")

import json
import re
from tqdm import tqdm
from datasets import load_dataset

# ====== 函数部分 ======

def extract_model_answer(response_text):
    """
    从模型响应中提取最终数值/答案，适配若干常见格式。
    """
    cleaned_response = response_text.replace("<|begin_of_text|>", "").strip()
    cleaned_response = cleaned_response.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
    cleaned_response = cleaned_response.replace("<|eot_id|>", "").strip()
    cleaned_response = cleaned_response.replace("</s>", "").strip()
    cleaned_response = cleaned_response.replace("<|end_of_text|>", "").strip()
    cleaned_response = cleaned_response.replace("<|end_of_sentence|>", "").strip()
    cleaned_response = re.sub(r'<\|.*?\|>', '', cleaned_response)
    cleaned_response = re.sub(r'\s+', ' ', cleaned_response).strip()

    if not cleaned_response:
        return ""

    patterns = [
        r'Answer\s*[:\.\s]*([^\n]+)',
        r'####\s*([^\n]+)',
        r'\\boxed{(.+?)}',
        r'boxed{(.+?)}',
        r'The answer is\s*[:\.\s]*([^\.\n]+)',
        r'Final Answer\s*[:\.\s]*([^\.\n]+)',
        r'Therefore\s*,\s*([^\.\n]+)',
        r'答案[:：]\s*([^\n]+)',
        r'所以[:：]\s*([^\n]+)',
        r'final answer\s*[:\.\s]*([^\.\n]+)',
        r'(\d+(\.\d+)?)',
        r'<answer>\s*(.*?)\s*</answer>',
        r'<answer>(.*?)</answer>'
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned_response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            number_match = re.search(r'([+-]?\d+(?:\.\d+)?)', answer)
            if number_match:
                return number_match.group(1)
            if answer:
                return answer

    lines = cleaned_response.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.lower().startswith(('question:', 'step', 'solution:', 'answer:', '问题:', '步骤:')):
            number_match = re.search(r'([+-]?\d+(?:\.\d+)?)$', line)
            if number_match:
                return number_match.group(1)

    number_match = re.search(r'([+-]?\d+(?:\.\d+)?)', cleaned_response)
    if number_match:
        return number_match.group(1)
    return ""

def is_correct_answer(predicted, true_answer):
    """
    判断预测答案与真值是否一致（针对MATH500数据集）
    """
    if not predicted or not true_answer:
        return False
    
    # 清理答案，移除多余空格
    pred_clean = re.sub(r'\s+', ' ', str(predicted)).strip()
    true_clean = re.sub(r'\s+', ' ', str(true_answer)).strip()
    
    # 直接比较（忽略空格和换行符）
    if pred_clean.replace(' ', '') == true_clean.replace(' ', ''):
        return True
    
    # 尝试数值比较（如果可能）
    try:
        pred_float = float(pred_clean.replace(',', ''))
        true_float = float(true_clean.replace(',', ''))
        return abs(pred_float - true_float) < 1e-6
    except ValueError:
        pass
    
    return False


# ====== 主评估函数 ======

def evaluate_math500(eval_file):
    """按顺序对齐 HuggingFaceH4/MATH-500 与你的模型输出"""
    # 1. 载入标准数据集（test split）
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # 2. 读取模型预测
    with open(eval_file, "r", encoding="utf-8") as f:
        preds = [json.loads(line) for line in f]

    assert len(preds) == len(dataset), \
        f"样本数不匹配！预测 {len(preds)} 条，但 MATH500 有 {len(dataset)} 条"

    # 3. 顺序比较
    correct = 0
    total = len(dataset)

    for i in tqdm(range(total), desc="Evaluating"):
        gt_item = dataset[i]
        pred_item = preds[i]

        true_ans = extract_model_answer(gt_item["solution"])
        pred_ans = extract_model_answer(pred_item.get("answer", ""))

        if is_correct_answer(pred_ans, true_ans):
            correct += 1

    acc = correct / total
    print(f"✅ Accuracy: {acc:.4f} ({correct}/{total})")


# ====== 执行 ======
evaluate_math500("Bo4_self_certainty.jsonl")

