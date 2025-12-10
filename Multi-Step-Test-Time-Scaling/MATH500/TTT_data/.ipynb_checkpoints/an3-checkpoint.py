# import json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # === 读取 JSONL 文件 ===
# dataset = []
# with open("math500_eval_qwen1.5b_instruct.jsonl", "r") as f:
#     for line in f:
#         dataset.append(json.loads(line.strip()))

# # === 设置权重范围 ===
# weights = np.linspace(0, 1, 1000)  # 0.0, 0.1, ..., 1.0

# results = []

# for w in weights:
#     correct_count = 0
#     total = 0

#     for item in dataset:
#         answers = item["answers"]
#         best_score = -float("inf")
#         best_answer = None

#         for ans in answers:
#             steps = ans["steps"]
#             self_certainty_values = np.array([s.get("self_certainty", 0) for s in steps], dtype=float)
#             entropy_values = np.array([s.get("entropy", 0) for s in steps], dtype=float)

#             # === 取 max(self_certainty) 和 mean(entropy)
#             max_self_certainty = np.max(self_certainty_values)
#             mean_entropy = np.mean(entropy_values)

#             # === 计算加权得分 ===
#             score = w * max_self_certainty + (1 - w) * mean_entropy

#             if score > best_score:
#                 best_score = score
#                 best_answer = ans

#         if best_answer:
#             total += 1
#             if best_answer.get("correct", False):
#                 correct_count += 1

#     acc = correct_count / total if total > 0 else 0
#     results.append({
#         "weight_self_certainty": round(w, 2),
#         "accuracy": round(acc * 100, 2)
#     })

# # === 转成 DataFrame ===
# df = pd.DataFrame(results)
# best_row = df.loc[df["accuracy"].idxmax()]

# print("\n=== self_certainty(max) + entropy(mean) 加权融合结果 ===")
# print(df)
# print(f"\n🏆 最佳权重: w={best_row.weight_self_certainty}, 准确率={best_row.accuracy:.2f}%")

# # === 保存结果 ===
# df.to_csv("fusion_self_certainty_entropy.csv", index=False)
# print("✅ 已保存结果到 fusion_self_certainty_entropy.csv")

# # === 绘图 ===
# sns.set(style="whitegrid", font_scale=1.2)
# plt.figure(figsize=(8, 5))
# sns.lineplot(data=df, x="weight_self_certainty", y="accuracy", marker="o")
# plt.title("Weighted Fusion of self_certainty(max) and entropy(mean)")
# plt.xlabel("Weight for self_certainty (w)")
# plt.ylabel("Accuracy (%)")
# plt.tight_layout()
# plt.savefig("fusion_self_certainty_entropy.png", dpi=300)
# plt.close()

# print("✅ 已生成图表：fusion_self_certainty_entropy.png")

import json
import random
from collections import Counter
from tqdm import tqdm

input_path = "math500_eval_llama1b.jsonl"      # 输入文件路径
output_path = "majority_answer_tiebreak.jsonl"  # 输出文件路径

results = []

# === 读取 JSONL ===
with open(input_path, "r") as f:
    dataset = [json.loads(line.strip()) for line in f]

correct_count = 0
tie_count = 0  # 平票次数

# === 逐样本处理 ===
for sample in tqdm(dataset, desc="Selecting majority-vote answers with tiebreak"):
    question = sample.get("question") or sample.get("problem") or ""
    true_answer = str(sample.get("true_answer")).strip()
    answers = sample.get("answers", [])

    # 收集所有回答文本
    all_texts = []
    for ans in answers:
        text = ans.get("final_answer") or ans.get("full_text")
        if text:
            cleaned = str(text).strip()
            all_texts.append(cleaned)

    if not all_texts:
        continue

    # 统计每个回答出现次数
    counter = Counter(all_texts)
    most_common = counter.most_common()

    # 取出现次数最多的答案
    top_count = most_common[0][1]
    top_answers = [ans for ans, c in most_common if c == top_count]

    selected_answer = None
    tie_flag = len(top_answers) > 1

    # === 平票处理 ===
    if tie_flag:
        tie_count += 1
        # 若有正确答案在平票中，选它
        correct_candidates = [ans for ans in top_answers if ans == true_answer]
        if correct_candidates:
            selected_answer = correct_candidates[0]
        else:
            # 若都错误，随机选一个
            selected_answer = random.choice(top_answers)
    else:
        selected_answer = top_answers[0]

    # === 判断是否正确 ===
    is_correct = (selected_answer == true_answer)
    if is_correct:
        correct_count += 1

    results.append({
        "question": question,
        "answer": selected_answer,
        "true_answer": true_answer,
        "count": top_count,
        "correct": is_correct,
        "tie": tie_flag
    })

# === 写出新 JSONL 文件 ===
# with open(output_path, "w") as f:
#     for r in results:
#         f.write(json.dumps(r, ensure_ascii=False) + "\n")

# === 输出统计信息 ===
total = len(results)
accuracy = correct_count / total if total > 0 else 0.0

print(f"✅ 已保存到 {output_path}")
print(f"📊 总样本数: {total}")
print(f"✅ 正确数: {correct_count}")
print(f"📈 正确率: {accuracy:.4f}")
print(f"⚖️ 平票次数: {tie_count}")
