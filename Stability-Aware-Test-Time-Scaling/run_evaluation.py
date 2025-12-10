import json
import argparse
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from fractions import Fraction
import re

from math_utils import strip_string, extract_answer


# ============================================================
# ★ 新增：统一数值归一化（解决 142.0 != 142 问题）
# ============================================================
def normalize_number(text):
    """
    将各种数字形式统一成 canonical string:
    - "142.0" -> "142"
    - " 142 " -> "142"
    - "3/4" -> "0.75"
    - "3\\frac{1}{3}" -> "3.3333333333"
    - 非数字返回 None
    """
    if text is None:
        return None

    text = str(text).strip()

    # boxed
    m = re.search(r"\\boxed\{(.+?)\}", text)
    if m:
        text = m.group(1)

    # fraction  a\frac{b}{c}
    m = re.search(r"(\d+)\s*\\frac\{(\d+)\}\{(\d+)\}", text)
    if m:
        a, b, c = m.groups()
        val = Fraction(int(a)*int(c) + int(b), int(c))
        return str(float(val))

    # simple fraction a/b
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", text)
    if m:
        a, b = m.groups()
        return str(float(Fraction(int(a), int(b))))

    # normal number
    m = re.search(r"-?\d+(\.\d+)?", text)
    if m:
        val = m.group(0)
        if val.endswith(".0"):
            val = val[:-2]   # remove .0
        return val

    return None


# ============================================================
# 1. 更鲁棒的 GT 提取函数
# ============================================================
def safe_extract_gt(raw_gt):
    """
    支持 float / int / dict / None / boxed
    """

    if raw_gt is None:
        return None

    # number
    if isinstance(raw_gt, (int, float)):
        raw_gt = str(raw_gt)

    # dict
    if isinstance(raw_gt, dict):
        for key in ["answer", "value", "gold", "gold_answer"]:
            if key in raw_gt:
                return safe_extract_gt(raw_gt[key])
        raw_gt = str(raw_gt)

    raw_gt = str(raw_gt)

    # try boxed/normal extraction
    ext = extract_answer(raw_gt)
    if ext not in [None, ""]:
        ext = strip_string(ext)
    else:
        ext = strip_string(raw_gt)

    if ext == "":
        return None

    # ★ normalize number
    norm = normalize_number(ext)
    if norm is not None:
        return norm

    return ext  # fallback to raw cleaned string


# ============================================================
# 2. 主评估逻辑
# ============================================================
def eval_strategies(input_file):
    print(f"Loading data: {input_file}")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): 
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue
            
    strategies = [
        "Baseline: Pass@1",
        "Baseline: Consistency (SC)",
        "Weighted: Z-Std-Entropy",
        "Weighted: Z-Avg-Certainty",
        "Hybrid: Double Stability",
        "Combo: FilterTopK + W-StdTopK"
    ]
    
    correct_counts = defaultdict(int)
    total = 0
    
    debug_printed = 0
    
    print("Running evaluation...")
    
    for item in tqdm(data):
        responses = item.get("responses", [])
        if not responses:
            continue

        # ------------------------------
        # 提取 GT（严格数值化）
        # ------------------------------
        raw_gt = item.get("answer", None)
        gt = safe_extract_gt(raw_gt)
        if gt is None:
            continue

        ### ★ norm_gt 已经是 normalize_number 后的 string，安全用于比较
        norm_gt = gt

        # ------------------------------
        # 筛选有效 response
        # ------------------------------
        valid_responses = [r for r in responses if r.get('extracted_answer')]
        if not valid_responses:
            continue
        
        total += 1
        
        # stats for weights
        vals = defaultdict(list)
        for r in valid_responses:
            vals['std_ent'].append(r.get('std_topk_entropy', 0))
            vals['avg_cert'].append(r.get('avg_certainty', 0))
            vals['step_std'].append(r.get('step_std_entropy', 0))
            
        stats = {}
        for k in vals:
            stats[k] = {
                'm': np.mean(vals[k]), 
                's': np.std(vals[k]) + 1e-9,
                'p80': np.percentile(vals[k], 80)
            }
            
        # ------------------------------
        # 加权 voting
        # ------------------------------
        candidates = defaultdict(lambda: defaultdict(float))
        
        pool_filtered = [
            r for r in valid_responses 
            if r.get('std_topk_entropy', 999) <= stats['std_ent']['p80']
        ]
        if not pool_filtered:
            pool_filtered = valid_responses 
        
        # 收集各答案的加权分数
        for r in valid_responses:
            raw_ans = r.get('extracted_answer')
            clean_ans = strip_string(raw_ans)

            ### ★ normalize prediction
            norm_ans = normalize_number(clean_ans)
            if norm_ans is None:
                continue

            z_std = -(r.get('std_topk_entropy', 0) - stats['std_ent']['m']) / stats['std_ent']['s']
            z_step_std = -(r.get('step_std_entropy', 0) - stats['step_std']['m']) / stats['step_std']['s']
            z_cert =  (r.get('avg_certainty', 0) - stats['avg_cert']['m']) / stats['avg_cert']['s']
            
            candidates[norm_ans]['count'] += 1
            candidates[norm_ans]['w_std'] += np.exp(z_std)
            candidates[norm_ans]['w_cert'] += np.exp(z_cert)
            candidates[norm_ans]['w_double'] += np.exp(z_std + z_step_std)
            
        # FilterTopK stage
        combo_votes = defaultdict(float)
        for r in pool_filtered:
            raw_ans = r.get("extracted_answer")
            clean_ans = strip_string(raw_ans)
            norm_ans = normalize_number(clean_ans)
            if norm_ans is None:
                continue

            z = -(r.get('std_topk_entropy', 0) - stats['std_ent']['m']) / stats['std_ent']['s']
            combo_votes[norm_ans] += np.exp(z)
            

        # ------------------------------
        # 逐策略判定
        # ------------------------------

        # 1 Pass@1
        p1_raw = valid_responses[0].get("extracted_answer")
        p1_norm = normalize_number(strip_string(p1_raw))
        if p1_norm == norm_gt:
            correct_counts["Baseline: Pass@1"] += 1

        # 2 SC voting
        if len(candidates) > 0:
            best_sc = max(candidates.items(), key=lambda x: x[1]['count'])[0]
            if best_sc == norm_gt:
                correct_counts["Baseline: Consistency (SC)"] += 1

        # 3 Z-std
        if len(candidates) > 0:
            best_std = max(candidates.items(), key=lambda x: x[1]['w_std'])[0]
            if best_std == norm_gt:
                correct_counts["Weighted: Z-Std-Entropy"] += 1
        
        # 4 Z-cert
        if len(candidates) > 0:
            best_cert = max(candidates.items(), key=lambda x: x[1]['w_cert'])[0]
            if best_cert == norm_gt:
                correct_counts["Weighted: Z-Avg-Certainty"] += 1
        
        # 5 Double stability
        if len(candidates) > 0:
            best_dbl = max(candidates.items(), key=lambda x: x[1]['w_double'])[0]
            if best_dbl == norm_gt:
                correct_counts["Hybrid: Double Stability"] += 1
        
        # 6 Combo
        if len(combo_votes) > 0:
            best_combo = max(combo_votes.items(), key=lambda x: x[1])[0]
            if best_combo == norm_gt:
                correct_counts["Combo: FilterTopK + W-StdTopK"] += 1


        # ------------------------------
        # Debug
        # ------------------------------
        if debug_printed < 10:
            print("\n[DEBUG]")
            print("GT Raw:", raw_gt)
            print("GT Extracted:", gt)
            print("norm_gt:", norm_gt)
            print("Pass@1 Pred Raw:", p1_raw)
            print("Pass@1 Pred Norm:", p1_norm)
            debug_printed += 1


    # ------------------------------
    # 输出报表
    # ------------------------------
    print("\n" + "="*80)
    print(f"{'GENERALIZATION BENCHMARK REPORT':^80}")
    print("="*80)
    print(f"{'Strategy':<35} | {'Accuracy':<10} | {'vs SC':<10}")
    print("-" * 80)
    
    base = correct_counts["Baseline: Consistency (SC)"] / total if total > 0 else 0
    
    sorted_res = sorted(correct_counts.items(), key=lambda x: x[1], reverse=True)
    
    for name, count in sorted_res:
        acc = count / total if total > 0 else 0
        gain = acc - base
        mark = "👑" if count == sorted_res[0][1] and count > 0 else ""
        print(f"{name:<35} | {acc:.2%}   | {gain:+.2%} {mark}")
        
    print("="*80)
    print(f"Total Samples: {total}")


# ============================================================
# 3. CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    
    eval_strategies(args.input_file)
