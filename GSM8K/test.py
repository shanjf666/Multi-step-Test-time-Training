import json
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

# Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 加载 JSON 文件
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    confidences = np.array([entry['max_confidence'] for entry in data['results']])
    labels = np.array([1 if entry['correct'] else 0 for entry in data['results']])
    return confidences, labels

# 计算 ECE (Expected Calibration Error)
def calculate_ece(probabilities, labels, n_bins=10):
    prob_true, prob_pred = calibration_curve(labels, probabilities, n_bins=n_bins, strategy='uniform')
    bin_sizes = np.histogram(probabilities, bins=np.linspace(0, 1, n_bins+1))[0]
    bin_weights = bin_sizes / len(probabilities)
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece

# 主函数
def main(file_path):
    confidences, labels = load_data(file_path)
    
    # 标准化 + Sigmoid 归一化
    standardized_conf = (confidences - np.mean(confidences)) / np.std(confidences)
    normalized_conf = sigmoid(standardized_conf)
    
    # 计算 Brier Score
    brier = brier_score_loss(labels, normalized_conf)
    
    # 计算 ECE
    ece = calculate_ece(normalized_conf, labels, n_bins=10)
    
    print(f"Brier Score: {brier:.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    
    return brier, ece

# 示例用法
if __name__ == "__main__":
    file_path = 'TTT_data/Best_of_4_Transformers_Step_Certainty_lambda_0.5_deepseek_7b_key_2.json'  # 替换为您的 JSON 文件路径
    main(file_path)