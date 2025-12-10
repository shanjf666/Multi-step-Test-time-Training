import json
import numpy as np
import pandas as pd
import matplotlib
# 设置无头模式，防止在服务器上报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import roc_auc_score

# 引入你的工具库
from math_utils import is_correct, extract_answer, clean_latex_format, strip_string

def get_majority_answer(responses):
    """
    获取多数投票选出的答案（经过严格归一化清洗）。
    防止因为格式不同（如 "42" vs "42.0"）导致票数分散，
    也防止垃圾字符（如 "."）成为多数派。
    """
    normalized_answers = []
    
    for r in responses:
        # 1. 获取原始提取结果
        raw_ans = r.get('extracted_answer', "")
        if not raw_ans:
            raw_ans = extract_answer(r.get('text', ""))
        
        # 2. [关键修复] 归一化
        # 移除 Latex 包装、空格、单位等，确保比较的一致性
        norm_ans = strip_string(clean_latex_format(raw_ans))
        
        # 3. 过滤无效答案
        # 如果归一化后是空字符串（说明原始只是标点或空格），不参与投票
        if norm_ans: 
            normalized_answers.append(norm_ans)

    if not normalized_answers:
        return None
    
    # 4. 统计票数
    c = Counter(normalized_answers)
    
    # 取票数最高的一个 [(answer, count)]
    major_ans_norm, count = c.most_common(1)[0]
    
    return major_ans_norm

def load_and_analyze_residuals(file_path):
    data = []
    print(f"Loading {file_path} for Residual Analysis...")
    
    stats = {
        "total_problems": 0,
        "problems_with_residuals": 0,
        "problems_dropped": 0 # 记录因为没有有效答案被丢弃的题目
    }

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            try:
                item = json.loads(line)
            except: 
                continue
            
            gt_answer = item['answer']
            responses = item['responses']
            
            stats["total_problems"] += 1
            
            # 1. 确定该题目的 Majority Answer (归一化后的)
            major_ans_norm = get_majority_answer(responses)
            
            if major_ans_norm is None: 
                stats["problems_dropped"] += 1
                continue

            has_residual = False
            
            for resp in responses:
                if 'error' in resp: continue
                
                # 获取当前样本的答案
                raw_ans = resp.get('extracted_answer', "")
                if not raw_ans: 
                    raw_ans = extract_answer(resp.get('text', ""))
                
                # [关键修复] 必须对当前答案也做同样的归一化，才能和 Majority 比较
                curr_ans_norm = strip_string(clean_latex_format(raw_ans))
                
                # 如果当前答案归一化后是空的，我们跳过这个样本，或者视为错误
                if not curr_ans_norm:
                    continue

                # 判断是否属于多数派
                is_majority = (curr_ans_norm == major_ans_norm)
                
                if not is_majority:
                    has_residual = True
                
                # 判断正误 (is_correct 内部也会做归一化比较)
                correct = is_correct(raw_ans, gt_answer)
                
                data.append({
                    "type": "Majority" if is_majority else "Residual",
                    "avg_entropy": resp.get('avg_topk_entropy', 0),
                    "std_entropy": resp.get('std_topk_entropy', 0), # 顺便收集标准差
                    "is_correct": correct,
                    "problem_id": stats["total_problems"]
                })
            
            if has_residual:
                stats["problems_with_residuals"] += 1
                
    print(f"Processed {stats['total_problems']} problems.")
    print(f"Problems with valid majority vote: {stats['total_problems'] - stats['problems_dropped']}")
    print(f"Problems containing deviation (residuals): {stats['problems_with_residuals']}")
    
    return pd.DataFrame(data)

def plot_residual_analysis(df):
    sns.set_theme(style="whitegrid", context="talk")
    
    # 1. 准备数据子集
    df_resid = df[df['type'] == "Residual"]
    df_major = df[df['type'] == "Majority"]
    
    # --- 图表 1: 熵值分布对比 (Majority vs Residual) ---
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_major, x="avg_entropy", label="Majority Answers", fill=True, color="#2ecc71", alpha=0.3)
    sns.kdeplot(data=df_resid, x="avg_entropy", label="Residual Answers", fill=True, color="#e74c3c", alpha=0.3)
    plt.title("Entropy Distribution: Majority vs. Residual")
    plt.xlabel("Average Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("resid_1_distribution_compare.png")
    print("Saved resid_1_distribution_compare.png")
    plt.close()

    # --- 图表 2: 剩余回答内部的 正确性 vs 熵 ---
    if len(df_resid) > 0:
        plt.figure(figsize=(10, 6))
        # 过滤掉极端的熵值以便更好观察
        plot_data = df_resid[df_resid['avg_entropy'] < 2.0] 
        sns.histplot(
            data=plot_data, x="avg_entropy", hue="is_correct",
            kde=True, bins=30,
            palette={True: "blue", False: "gray"}, 
            alpha=0.6, element="step"
        )
        plt.title("Residuals Only: Entropy vs Correctness")
        plt.xlabel("Average Entropy (Uncertainty)")
        plt.tight_layout()
        plt.savefig("resid_2_correctness.png")
        print("Saved resid_2_correctness.png")
        plt.close()

    # --- 图表 3: 准确率曲线 (Calibration Comparison) ---
    # 对比 Majority 组和 Residual 组，准确率随熵值的变化
    try:
        df['entropy_bin'] = pd.cut(df['avg_entropy'], bins=8)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df, x="avg_entropy", y="is_correct", hue="type",
            style="type", markers=True, dashes=False, err_style="bars",
            palette={"Majority": "#2ecc71", "Residual": "#e74c3c"}
        )
        plt.title("Accuracy vs Entropy: Group Comparison")
        plt.xlabel("Average Entropy")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("resid_3_calibration_compare.png")
        print("Saved resid_3_calibration_compare.png")
        plt.close()
    except Exception as e:
        print(f"Skipping plot 3 due to error: {e}")

def print_stats(df):
    print("\n" + "="*40)
    print("RESIDUAL ANALYSIS SUMMARY (NORMALIZED)")
    print("="*40)
    
    df_resid = df[df['type'] == "Residual"]
    df_major = df[df['type'] == "Majority"]
    
    print(f"Count - Majority: {len(df_major)}, Residual: {len(df_resid)}")
    print(f"Accuracy - Majority: {df_major['is_correct'].mean():.2%}")
    print(f"Accuracy - Residual: {df_resid['is_correct'].mean():.2%}")
    
    print("-" * 30)
    print(f"Avg Entropy - Majority: {df_major['avg_entropy'].mean():.4f}")
    print(f"Avg Entropy - Residual: {df_resid['avg_entropy'].mean():.4f}")
    
    print("-" * 30)
    # 计算区分度 AUROC
    if len(df_resid) > 0 and df_resid['is_correct'].nunique() > 1:
        auc = roc_auc_score(df_resid['is_correct'], -df_resid['avg_entropy'])
        print(f"AUROC (Residuals only): {auc:.4f}")
    else:
        print("AUROC (Residuals): Cannot calculate (single class present)")
        
    if len(df_major) > 0 and df_major['is_correct'].nunique() > 1:
        auc_maj = roc_auc_score(df_major['is_correct'], -df_major['avg_entropy'])
        print(f"AUROC (Majority only):  {auc_maj:.4f}")

if __name__ == "__main__":
    # 替换你的文件名
    FILE_PATH = "benchmark_results_multi/Qwen_Qwen2.5-Math-1.5B_AI-MO_aimo-validation-amc_20251209_114831.jsonl"
    
    df = load_and_analyze_residuals(FILE_PATH)
    
    if not df.empty:
        print_stats(df)
        plot_residual_analysis(df)
    else:
        print("No valid data found.")