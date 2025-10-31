import json
import pandas as pd
import numpy as np

# 读取 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['results'], data['accuracy']

# 分析不同阈值下的准确率和保留比例
def analyze_thresholds(file_path, thresholds=None):
    # 加载数据
    results, original_accuracy = load_json(file_path)
    
    # 如果没有指定阈值，生成 0.5 到 0.95 的等间隔阈值
    if thresholds is None:
        thresholds = np.arange(0.5, 1.0, 0.05)
    
    # 初始化统计结果
    stats = []
    total_samples = len(results)
    
    for threshold in thresholds:
        # 过滤样本
        filtered = [entry for entry in results if entry['max_confidence'] >= threshold]
        if not filtered:  # 避免空列表
            stats.append({
                'Threshold': round(threshold, 4),
                'Retained Samples': 0,
                'Retained Ratio': 0.0,
                'Accuracy': 0.0
            })
            continue
        
        # 计算过滤后的准确率和保留比例
        correct_count = sum(1 for entry in filtered if entry['correct'])
        retained_count = len(filtered)
        retained_ratio = retained_count / total_samples
        accuracy = correct_count / retained_count if retained_count > 0 else 0
        
        stats.append({
            'Threshold': round(threshold, 4),
            'Retained Samples': retained_count,
            'Retained Ratio': round(retained_ratio, 4),
            'Accuracy': round(accuracy, 4)
        })
    
    # 创建表格
    df = pd.DataFrame(stats)
    
    # 打印表格
    print(f"\nOriginal Accuracy: {original_accuracy:.4f}")
    print("\nThreshold Analysis Results:")
    print(df)
    
    return df, thresholds, original_accuracy

# 示例用法
file_path = 'TTT_data/Best_of_4_Transformers_Step_Certainty_lambda_0.5_gsm8k_filtered_step_reward_key.json'  # 替换为你的 JSON 文件路径
thresholds = np.arange(4.75, 6, 0.05)  # 阈值从 0.5 到 0.95，步长 0.05
df, thresholds, original_accuracy = analyze_thresholds(file_path, thresholds)