"""
evaluate-sc.py
python evaluate-sc.py --input_file qwen64.jsonl --output_hard_file question.jsonl
python evaluate-sc.py --input_file try64_1.jsonl --output_hard_file question_1.jsonl
python evaluate-sc.py --input_file llama64.jsonl --output_hard_file llamaquestion.jsonl
python evaluate-sc.py --input_file 50step64.jsonl --output_hard_file 50stepquestion.jsonl
python evaluate-sc.py --input_file scresults/select_32_step_80_bo64.jsonl --output_hard_file scresults/select_32_step_80_bo64question.jsonl
"""
import json
import argparse
import re
from typing import Optional

# =====================================================
# 1. 复用清洗工具函数 (保持与生成阶段一致)
# =====================================================

def strip_string(string):
    if string is None:
        return ""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace(" ", "")
    if string == "0.5": string = "\\frac{1}{2}"
    return string

def last_boxed_only_string(string):
    if not isinstance(string, str): return None
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

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
    if s is None:
        return None
    if "\\boxed " in s:
        return s.replace("\\boxed ", "")
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{") : -1]
    return s

def extract_answer(text: str) -> str:
    """
    通用提取函数：提取 \boxed{} 并标准化
    """
    boxed = last_boxed_only_string(text)
    if boxed is None:
        return None
    content = remove_boxed(boxed)
    return strip_string(content)

# =====================================================
# 2. 评估逻辑
# =====================================================

def evaluate_file(input_file, top_k, output_hard_file):
    total = 0
    correct = 0
    
    # 存储所有记录以便排序和分析
    all_records = []

    print(f"Processing {input_file} ...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        # 使用 enumerate 获取行号作为 idx
        for idx, line in enumerate(f):
            line = line.strip()
            if not line: continue
            
            data = json.loads(line)
            total += 1
            
            # 1. 获取模型预测
            pred_ans = data.get("sc_answer") 
            
            # 2. 获取标准答案
            raw_gt = data.get("answer")
            
            # 尝试提取标准答案
            extracted_gt = extract_answer(raw_gt)
            
            if extracted_gt is None or extracted_gt == "":
                 norm_gt = strip_string(raw_gt)
            else:
                 norm_gt = extracted_gt

            # 3. 比较
            if pred_ans is None:
                is_match = False
            else:
                is_match = (pred_ans == norm_gt)
            
            if is_match:
                correct += 1
            
            # 4. 记录数据用于分析
            extracted_distribution = []
            for r in data.get("responses", []):
                e = extract_answer(r)
                extracted_distribution.append(e if e else "[None]")

            all_records.append({
                "idx": idx,  # 保存索引
                "problem": data.get("problem"),
                "sc_score": data.get("sc_score", 0.0),
                "pred": pred_ans,       
                "gt": norm_gt,
                "is_correct": is_match,
                "extracted_distribution": extracted_distribution
            })

    # =====================================================
    # 3. 输出基础统计结果
    # =====================================================
    overall_accuracy = 100 * correct / total if total > 0 else 0
    print("-" * 60)
    print(f"Total Samples : {total}")
    print(f"Overall Correct: {correct}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print("-" * 60)

    # =====================================================
    # 3.5 [分析 1] 累积阈值分析 (Accuracy for Score >= T)
    # =====================================================
    print("\n=== Analysis 1: Accuracy by Threshold (Score >= T) ===")
    print(f"{'Threshold':<15} | {'Samples':<10} | {'Correct':<10} | {'Accuracy':<10}")
    print("-" * 60)

    thresholds = [i/10.0 for i in range(11)] 
    for t in thresholds:
        subset = [r for r in all_records if r['sc_score'] >= t - 1e-9]
        subset_total = len(subset)
        subset_correct = sum(1 for r in subset if r['is_correct'])
        
        acc = (subset_correct / subset_total * 100) if subset_total > 0 else 0.0
        print(f"{t:<15.1f} | {subset_total:<10} | {subset_correct:<10} | {acc:<9.2f}%")
    print("-" * 60)

    # =====================================================
    # 3.6 [分析 2] 区间分析 (Accuracy for Low <= Score < High)
    # =====================================================
    print("\n=== Analysis 2: Accuracy by Interval (Bin Calibration) ===")
    print(f"{'Interval':<15} | {'Samples':<10} | {'Correct':<10} | {'Accuracy':<10}")
    print("-" * 60)

    step = 0.1
    for i in range(10):
        low = i * step
        high = (i + 1) * step
        
        # 处理最后一个区间，使其包含 1.0 -> [0.9, 1.0]
        # 其他区间为左闭右开 -> [0.0, 0.1), [0.1, 0.2) ...
        if i == 9:
            label = f"[{low:.1f}, {high:.1f}]"
            subset = [r for r in all_records if low <= r['sc_score'] <= high + 1e-9]
        else:
            label = f"[{low:.1f}, {high:.1f})"
            subset = [r for r in all_records if low <= r['sc_score'] < high]
            
        subset_total = len(subset)
        subset_correct = sum(1 for r in subset if r['is_correct'])
        acc = (subset_correct / subset_total * 100) if subset_total > 0 else 0.0
        
        print(f"{label:<15} | {subset_total:<10} | {subset_correct:<10} | {acc:<9.2f}%")
    print("-" * 60)

    # =====================================================
    # 4. 找出 SC Score 最低的 Top K 并保存
    # =====================================================
    # 按 sc_score 升序排序
    sorted_records = sorted(all_records, key=lambda x: x['sc_score'])
    hardest_problems = sorted_records[:top_k]

    print(f"\n=== Top {top_k} Most Uncertain Problems (Lowest SC Score) ===")
    for i, rec in enumerate(hardest_problems):
        print(f"\n[Rank {i+1}] idx: {rec['idx']} | SC Score: {rec['sc_score']:.2f} | Correct: {rec['is_correct']}")
        print(f"Problem: {rec['problem'][:100]}..." if len(rec['problem']) > 100 else f"Problem: {rec['problem']}")
        print(f"Model Prediction   : {rec['pred']}")
        print(f"Distribution: {rec['extracted_distribution']}")
        print("-" * 30)

    # 保存到文件
    if output_hard_file:
        print(f"\nSaving top {top_k} hardest problems to: {output_hard_file}")
        with open(output_hard_file, "w", encoding="utf-8") as f_out:
            for rec in hardest_problems:
                # --- 修改处：按要求字段名保存 ---
                out_record = {
                    "idx": rec["idx"],
                    "question": rec["problem"],
                    "best_answer": rec["pred"],
                    "score_norm": rec["sc_score"],
                    "correct": rec["is_correct"]
                }
                f_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")
        print("Save completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate math predictions and save low-confidence examples.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the .jsonl file generated by generate.py")
    parser.add_argument("--top_k", type=int, default=32, help="Number of low-confidence examples to process")
    parser.add_argument("--output_hard_file", type=str, default="hard_problems.jsonl", help="File path to save the top k hard problems")
    
    args = parser.parse_args()
    
    evaluate_file(args.input_file, args.top_k, args.output_hard_file)