"""
python filter_difficulty.py --input_file qwen64.jsonl --output_dir grade
"""
import json
import argparse
import os
import sys
from tqdm import tqdm

# ==========================================
# 1. 基础提取工具 (用于匹配回答)
# ==========================================
def last_boxed_only_string(string):
    if not isinstance(string, str): return None
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0: return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]
    return retval

def remove_boxed(s):
    if s is None: return None
    if "\\boxed " in s: return s.replace("\\boxed ", "")
    if s.startswith("\\boxed{") and s.endswith("}"): return s[len("\\boxed{") : -1]
    return s

def strip_string(string):
    if string is None: return ""
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\$", "")
    string = string.replace(" ", "")
    if string == "0.5": string = "\\frac{1}{2}"
    return string

def extract_normalized_answer(text):
    """提取并标准化模型输出中的答案"""
    return strip_string(remove_boxed(last_boxed_only_string(text)))

# ==========================================
# 2. 纯一致性筛选主逻辑
# ==========================================
def process_consistency_only(input_file, output_dir, high_thresh, low_thresh):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 数据桶
    buckets = {
        "easy": [],
        "medium": [],
        "hard": []
    }
    
    # 统计
    stats = {"easy": 0, "medium": 0, "hard": 0, "skipped": 0}

    print(f"Reading from: {input_file}...")
    print(f"Criteria: Easy >= {high_thresh}, Hard < {low_thresh}")
    print("NOTE: Ignoring Ground Truth correctness. Selecting majority vote answer.")

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines):
        if not line.strip(): continue
        try:
            data = json.loads(line)
        except:
            continue

        prompt = data.get("problem")
        gt_answer = data.get("answer") # 保留它只是为了输出格式，不参与筛选
        sc_score = data.get("sc_score", 0.0)
        sc_answer = data.get("sc_answer") #这是 Self-Consistency 算出来的“共识答案”
        responses = data.get("responses", [])

        # --- 核心逻辑：寻找对应 sc_answer 的那条路径 ---
        # 我们不管 sc_answer 对不对，只要它是一致性最高的，我们就选生成它的那条路径
        
        selected_solution = None
        
        if sc_answer is None:
            # 如果连 sc_answer 都没算出来（比如所有回复都无法解析），则跳过
            stats["skipped"] += 1
            continue

        target_ans_norm = strip_string(sc_answer)
        
        for resp in responses:
            # 提取当前这条回复的答案
            resp_ans_norm = extract_normalized_answer(resp)
            # 如果这条回复的答案 等于 共识答案，就选它
            if resp_ans_norm == target_ans_norm:
                selected_solution = resp
                break
        
        # 如果没找到匹配的（理论上不应该发生，除非提取逻辑不一致），选第一条或者跳过
        # 这里为了稳健，如果没找到精确匹配，选第一条作为 fallback
        if selected_solution is None and responses:
             selected_solution = responses[0]
        elif selected_solution is None:
             stats["skipped"] += 1
             continue

        # --- 构造输出对象 ---
        out_record = {
            "prompt": prompt,
            "answer": gt_answer,      # 真实答案 (仅存储)
            "solution": selected_solution # 模型的一致性推理路径 (不一定正确，但最自信)
        }

        # --- 纯一致性分流 ---
        if sc_score >= high_thresh:
            buckets["easy"].append(out_record)
            stats["easy"] += 1
        elif sc_score < low_thresh:
            buckets["hard"].append(out_record)
            stats["hard"] += 1
        else:
            buckets["medium"].append(out_record)
            stats["medium"] += 1

    # --- 写入文件 ---
    print("\nWriting JSON files...")
    for key, data_list in buckets.items():
        out_path = os.path.join(output_dir, f"rl_consistency_{key}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)
        print(f"Saved {key}: {len(data_list)} samples -> {out_path}")
    
    print(f"\nStats: {stats}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="rl_consistency_data")
    # 可以根据采样次数 N 调整阈值。如果 N=4，High建议0.75以上，Low建议0.5以下
    parser.add_argument("--high_threshold", type=float, default=0.8, help="Score >= this is Easy")
    parser.add_argument("--low_threshold", type=float, default=0.5, help="Score < this is Hard")
    args = parser.parse_args()
    
    process_consistency_only(args.input_file, args.output_dir, args.high_threshold, args.low_threshold)