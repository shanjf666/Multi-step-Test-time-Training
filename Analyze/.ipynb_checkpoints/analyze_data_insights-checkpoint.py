"""
python analyze_data_insights.py --input_file analyze.jsonl
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
def eval_topk_strategies(input_file):
    print(f"Reading data from: {input_file}")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): data.append(json.loads(line))

    print(f"Evaluating Top-K Entropy Strategies on FULL sample set...")

    strategy_correct = defaultdict(int)
    total_samples = 0
    
    for item in tqdm(data):
        all_responses = item.get("responses", [])
        valid_responses = [r for r in all_responses if r.get('extracted_answer')]
        if not valid_responses: continue
        
        gt = item.get("answer", "")
        norm_gt = strip_string(gt)
        total_samples += 1

        # --- 0. 预处理：基于 Top-K Entropy 计算 Step 聚合指标 ---
        for r in valid_responses:
            steps = r.get('steps', [])
            
            # 检查是否存在 Top-K 字段，如果不存在说明没跑 append 脚本，用普通 entropy 兜底
            has_topk = 'avg_topk_entropy' in r
            ent_key = 'avg_topk_entropy' if has_topk else 'avg_entropy'
            
            # 步骤级 TopK 聚合
            if not steps:
                r['_max_step_topk'] = r.get(ent_key, 999)
                r['_mean_step_topk'] = r.get(ent_key, 999)
                r['_std_step_topk'] = 0.0
            else:
                # 注意：步骤里的 key 也要对应
                s_ents = [s.get(ent_key, 999) for s in steps]
                r['_max_step_topk'] = max(s_ents) if s_ents else 999
                r['_mean_step_topk'] = np.mean(s_ents) if s_ents else 999
                r['_std_step_topk'] = np.std(s_ents) if s_ents else 0.0

        # --- 1. 计算统计分布 (Mean/Std) 用于 Z-Score ---
        vals = defaultdict(list)
        for r in valid_responses:
            # 使用 Top-K Entropy 指标
            # 如果 jsonl 里没有 avg_topk_entropy，这里就会用兜底逻辑，请确保使用了 append 后的文件
            topk_ent = r.get('avg_topk_entropy', r.get('avg_entropy', 999))
            std_topk = r.get('std_topk_entropy', r.get('std_entropy', 999))
            
            vals['avg_topk'].append(topk_ent)
            vals['std_topk'].append(std_topk)
            
            # Step Aggregations
            vals['max_step_topk'].append(r['_max_step_topk'])
            vals['std_step_topk'].append(r['_std_step_topk'])
            
            # 辅助指标 (Certainty) - 保持使用原版，因为 Certainty 计算逻辑通用
            vals['avg_cert'].append(r.get('avg_certainty', -999))

        stats = {}
        for key in vals:
            stats[key] = {'m': np.mean(vals[key]), 's': np.std(vals[key]) + 1e-9}
            stats[key]['p80'] = np.percentile(vals[key], 80) # 用于过滤高熵(差)

        # --- 2. 准备数据结构 ---
        candidates = defaultdict(lambda: defaultdict(float))
        
        for r in valid_responses:
            ans = strip_string(r.get('extracted_answer'))
            
            # 获取原始值
            v_topk = r.get('avg_topk_entropy', r.get('avg_entropy', 999))
            v_std_topk = r.get('std_topk_entropy', r.get('std_entropy', 999))
            v_cert = r.get('avg_certainty', -999)
            
            # --- 计算 Z-Scores (统一方向：越大越好) ---
            
            # A. Entropy 类 (越小越好 -> 取反)
            z_topk = -(v_topk - stats['avg_topk']['m']) / stats['avg_topk']['s']
            z_std_topk = -(v_std_topk - stats['std_topk']['m']) / stats['std_topk']['s']
            z_max_step_topk = -(r['_max_step_topk'] - stats['max_step_topk']['m']) / stats['max_step_topk']['s']
            z_std_step_topk = -(r['_std_step_topk'] - stats['std_step_topk']['m']) / stats['std_step_topk']['s']
            
            # B. Certainty 类 (越大越好)
            z_cert = (v_cert - stats['avg_cert']['m']) / stats['avg_cert']['s']

            # 保存 Z-Score 到对象 (用于 Combo)
            r['_z_std_topk'] = z_std_topk
            r['_z_double_stab_topk'] = z_std_topk + z_std_step_topk

            # --- 累加权重 ---
            candidates[ans]['count'] += 1 # SC Baseline
            
            # 1. 复刻季军: Weighted Z-Std (TopK Version)
            candidates[ans]['w_std_topk'] += np.exp(z_std_topk)
            
            # 2. 复刻亚军: Hybrid Double Stability (TopK Version)
            # 序列 TopK 稳定 + 步骤 TopK 稳定
            candidates[ans]['h_double_stab_topk'] += np.exp(z_std_topk + z_std_step_topk)
            
            # 3. 新增: Weighted Z-MaxStep (TopK Version)
            candidates[ans]['w_max_step_topk'] += np.exp(z_max_step_topk)
            
            # 4. 新增: Weighted Z-Avg (TopK Version)
            candidates[ans]['w_avg_topk'] += np.exp(z_topk)
            
            # 5. 新增: Hybrid TopK-Precision (Certainty + StdTopK)
            # 结合"确信度"和"TopK稳定性"
            candidates[ans]['h_prec_topk'] += np.exp(z_cert + z_std_topk)

        # =========================================
        # 3. 结算策略
        # =========================================
        
        # A. Baseline
        # Pass@1
        p1_ans = valid_responses[0].get('extracted_answer')
        if p1_ans and strip_string(p1_ans) == norm_gt: strategy_correct['Baseline: Pass@1'] += 1
        
        # SC
        best_sc = max(candidates.items(), key=lambda x: x[1]['count'])[0]
        if best_sc == norm_gt: strategy_correct['Baseline: SC'] += 1

        # B. Weighted Strategies
        w_map = [
            ('Weighted: Z-Std-TopK', 'w_std_topk'),
            ('Hybrid: Z_Std_TopK + Z_Std_Step_TopK', 'h_double_stab_topk'),
            ('Weighted: Z-MaxStep-TopK', 'w_max_step_topk'),
            ('Weighted: Z-Avg-TopK', 'w_avg_topk'),
            ('Hybrid: Z_Cert+Z_Std_TopK', 'h_prec_topk')
        ]
        for s_name, s_key in w_map:
            best_w = max(candidates.items(), key=lambda x: x[1][s_key])[0]
            if best_w == norm_gt: strategy_correct[s_name] += 1

        # C. Combo Strategy: 复刻冠军 (Filter Unstable TopK + W-StdTopK)
        # 过滤条件：std_topk_entropy <= p80 (保留稳定的)
        # 注意：这里是用原始值比较，std_topk_entropy 越小越好，所以保留 <= p80
        threshold = stats['std_topk']['p80']
        pool_combo = [r for r in valid_responses if r.get('std_topk_entropy', r.get('std_entropy', 999)) <= threshold]
        if not pool_combo: pool_combo = valid_responses
        
        combo_counts = defaultdict(float)
        for r in pool_combo:
            ans = strip_string(r.get('extracted_answer'))
            combo_counts[ans] += np.exp(r['_z_std_topk'])
            
        best_combo = max(combo_counts.items(), key=lambda x: x[1])[0]
        if best_combo == norm_gt: strategy_correct['Combo: FilterTopK + Z-StdTopK'] += 1

    # --- 输出报表 ---
    print("\n" + "="*95)
    print(f"{'TOP-K ENTROPY STRATEGY LEADERBOARD':^95}")
    print("="*95)
    print(f"{'Rank':<4} | {'Strategy':<45} | {'Accuracy':<10} | {'Gain vs SC':<10}")
    print("-" * 95)
    
    base_acc = strategy_correct['Baseline: SC'] / total_samples if total_samples > 0 else 0
    sorted_strats = sorted(strategy_correct.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (name, count) in enumerate(sorted_strats, 1):
        acc = count / total_samples if total_samples > 0 else 0
        gain = acc - base_acc
        
        gain_str = f"{gain:+.2%}"
        if 'Baseline' in name: gain_str = "-"

        
        print(f"{rank:<4} | {name:<45} | {acc:.2%}   | {gain_str:<10}")
        
    print("="*95)
    print(f"Total Samples Evaluated: {total_samples}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    
    eval_topk_strategies(args.input_file)