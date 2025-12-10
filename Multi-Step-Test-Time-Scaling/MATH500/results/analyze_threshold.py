"""
python analyze_threshold.py --input math500_scores.jsonl
"""
import json
import argparse
import numpy as np
from tabulate import tabulate

def load_scores(path):
    scores = []
    corrects = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            scores.append(obj["score_norm"])
            corrects.append(obj["correct"])
    return np.array(scores), np.array(corrects, dtype=bool)

def main(args):
    scores, corrects = load_scores(args.input)

    thresholds = np.arange(0.0, 1.01, 0.05)
    rows = []

    for t in thresholds:
        mask = scores <= t
        n = mask.sum()
        acc = corrects[mask].mean() if n > 0 else 0.0
        rows.append([f"{t:.2f}", n, f"{acc:.2%}"])

    print("\n=== 阈值与正确率关系表 ===\n")
    print(tabulate(rows, headers=["阈值 t", "样本数(<=t)", "正确率"], tablefmt="grid"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="scores.jsonl 文件路径")
    args = parser.parse_args()
    main(args)
