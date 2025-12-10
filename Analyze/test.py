"""
python test.py --input_file analyze.jsonl
"""
import json
import argparse
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

# -----------------------------------------------------------------------------
# 1. 基础工具
# -----------------------------------------------------------------------------
def strip_string(string):
    if string is None: return ""
    string = str(string)
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace(" ", "")
    if string == "0.5": string = "\\frac{1}{2}"
    return string

# -----------------------------------------------------------------------------
# 2. 核心评测逻辑
# -----------------------------------------------------------------------------
def process_file(input_file):
    print(f"Reading data from: {input_file}")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))
            
    # 初始化计数器
    strategies = [
        "Pass@1", 
        "SC (Majority)", 
        "Weighted: Length (1/len)", 
        "Weighted: Z-Certainty", 
        "Weighted: Z-Entropy",
        "Hybrid: Certainty / log(Len)",
        "Filter: Drop Longest 20%"
    ]
    correct_counts = defaultdict(int)
    total = 0

    print("Evaluating Strategies...")

    for item in tqdm(data):
        responses = item.get("responses", [])
        if not responses: continue
        
        # 1. 获取 GT
        gt = item.get("answer", "")
        # 如果原来文件里已经计算了 extraction 和 score，直接用；否则这里需要重新清洗 GT
        # 这里假设 responses[0]['extracted_answer'] 已经是清洗过的，直接对比字符串即可
        # 为了稳妥，我们还是做一次 strip
        norm_gt = strip_string(gt)

        # 2. 预计算该问题的统计特征 (用于 Z-Score)
        # 注意：这里需要处理 token_count 为 0 的情况
        stats = defaultdict(list)
        for r in responses:
            toks = r.get('token_count', 0)
            if toks == 0: toks = len(r.get('raw_text', '')) # 兜底
            r['_tokens'] = toks # 存回对象方便后续用
            
            stats['certainty'].append(r.get('avg_certainty', 0))
            stats['entropy'].append(r.get('avg_entropy', 0))
        
        # 计算均值和方差
        means = {k: np.mean(v) for k, v in stats.items()}
        stds = {k: np.std(v) + 1e-9 for k, v in stats.items()}

        # 3. 准备候选答案池
        # 结构: { "Answer_String": { "count": N, "score_A": ..., "score_B": ... } }
        candidates = defaultdict(lambda: defaultdict(float))
        
        # 4. 遍历 Responses 计算各种权重
        valid_responses_for_filter = [] # 用于 Filter 策略
        
        # 设定过滤阈值 (例如长度的 80 分位数)
        all_lens = [r['_tokens'] for r in responses]
        len_threshold = np.percentile(all_lens, 80) # 剔除最长的 20%

        for r in responses:
            ans = r.get('extracted_answer')
            if not ans: continue
            norm_ans = strip_string(ans)
            
            # --- 原始指标 ---
            toks = r['_tokens']
            cert = r.get('avg_certainty', 0)
            ent = r.get('avg_entropy', 0)
            
            # --- Z-Scores ---
            z_cert = (cert - means['certainty']) / stds['certainty']
            z_ent = (ent - means['entropy']) / stds['entropy']
            
            # === 策略权重计算 ===
            
            # Strategy 1: SC (Count)
            candidates[norm_ans]['sc'] += 1.0
            
            # Strategy 2: Length Penalty (依据你的 d=-0.48，这是强特征)
            # 使用倒数，越短分越高
            w_len = 1.0 / (toks + 1.0)
            candidates[norm_ans]['len'] += w_len
            
            # Strategy 3: Z-Certainty (依据 d=0.25)
            # Z可能为负，使用 exp(Z) 保证为正且放大差异
            # 或者简单的 Z 加分 (需要注意基数，这里用 sigmoid 变体或直接累加均可)
            # 这里用 Softmax 思想的 exp
            w_z_cert = np.exp(z_cert) 
            candidates[norm_ans]['z_cert'] += w_z_cert
            
            # Strategy 4: Z-Entropy (依据 d=-0.24)
            # 熵越低越好 -> Z越小越好 -> -Z 越大越好
            w_z_ent = np.exp(-z_ent)
            candidates[norm_ans]['z_ent'] += w_z_ent
            
            # Strategy 5: Hybrid (Insights Combo)
            # 结合 长度(强) 和 确定性(中)
            # 公式: exp(Z_Certainty) / log(Len)
            # 既奖励相对自信，又严厉惩罚啰嗦
            w_hybrid = np.exp(z_cert) / np.log(toks + 2.0)
            candidates[norm_ans]['hybrid'] += w_hybrid
            
            # Strategy 6: Filter Logic
            if toks <= len_threshold:
                valid_responses_for_filter.append(norm_ans)

        # 5. 结算
        total += 1
        
        # Pass@1
        p1_ans = responses[0].get('extracted_answer')
        if p1_ans and strip_string(p1_ans) == norm_gt:
            correct_counts["Pass@1"] += 1

        # SC
        best_sc = max(candidates.items(), key=lambda x: x[1]['sc'])[0]
        if best_sc == norm_gt: correct_counts["SC (Majority)"] += 1
        
        # Weighted Length
        best_len = max(candidates.items(), key=lambda x: x[1]['len'])[0]
        if best_len == norm_gt: correct_counts["Weighted: Length (1/len)"] += 1
        
        # Weighted Z-Cert
        best_z_cert = max(candidates.items(), key=lambda x: x[1]['z_cert'])[0]
        if best_z_cert == norm_gt: correct_counts["Weighted: Z-Certainty"] += 1
        
        # Weighted Z-Ent
        best_z_ent = max(candidates.items(), key=lambda x: x[1]['z_ent'])[0]
        if best_z_ent == norm_gt: correct_counts["Weighted: Z-Entropy"] += 1
        
        # Hybrid
        best_hybrid = max(candidates.items(), key=lambda x: x[1]['hybrid'])[0]
        if best_hybrid == norm_gt: correct_counts["Hybrid: Certainty / log(Len)"] += 1
        
        # Filter + SC
        if valid_responses_for_filter:
            c = Counter(valid_responses_for_filter)
            best_filter = c.most_common(1)[0][0]
            if best_filter == norm_gt: correct_counts["Filter: Drop Longest 20%"] += 1
        else:
            # 如果全被过滤了（极端情况），回退到 SC
            if best_sc == norm_gt: correct_counts["Filter: Drop Longest 20%"] += 1

    # --- 输出最终报表 ---
    print("\n" + "="*80)
    print(f"{'FINAL STRATEGY EVALUATION':^80}")
    print("="*80)
    print(f"{'Strategy Name':<35} | {'Accuracy':<10} | {'Gain vs SC':<10}")
    print("-" * 80)
    
    base_acc = correct_counts["SC (Majority)"] / total
    
    # 按准确率排序输出
    sorted_stats = sorted(correct_counts.items(), key=lambda x: x[1], reverse=True)
    
    for name, count in sorted_stats:
        acc = count / total
        diff = acc - base_acc
        print(f"{name:<35} | {acc:.2%}   | {diff:+.2%}")
        
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    
    process_file(args.input_file)