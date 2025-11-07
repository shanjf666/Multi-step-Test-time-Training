import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 读取 JSONL 文件 ===
dataset = []
with open("math500_eval.jsonl", "r") as f:
    for line in f:
        dataset.append(json.loads(line.strip()))

# === 要分析的指标 ===
metrics = ["step_probability", "self_certainty", "entropy", "self_eval", "coe_c"]

# === 全局绘图参数 ===
sns.set(style="whitegrid", font_scale=1.2)

for metric in metrics:
    data_list = []

    for item in dataset:
        answers = item["answers"]

        # 选当前指标平均值最高的回答
        best_answer = None
        best_score = -float("inf")

        for answer in answers:
            steps = answer["steps"]
            mean_metric = np.mean([s.get(metric, 0) for s in steps])
            if mean_metric > best_score:
                best_score = mean_metric
                best_answer = answer

        # 记录该问题的最佳回答及其得分
        steps = best_answer["steps"]
        data_list.append({
            "correct": str(best_answer["correct"]),  # 转为字符串便于绘图
            f"mean_{metric}": np.mean([s.get(metric, 0) for s in steps])
        })

    df = pd.DataFrame(data_list)
    correct_col = "correct"
    metric_col = f"mean_{metric}"

    # === 图1：箱线图（Boxplot） ===
    plt.figure(figsize=(8,5))
    sns.boxplot(x=correct_col, y=metric_col, data=df,
                palette={"True": "green", "False": "red"})
    sns.stripplot(x=correct_col, y=metric_col, data=df,
                  color="black", alpha=0.3, jitter=True)
    plt.title(f"[Boxplot] {metric} vs Correctness (Best Answer per Sample)")
    plt.xlabel("Correctness (True=Correct, False=Incorrect)")
    plt.ylabel(f"Mean {metric}")
    plt.tight_layout()
    plt.savefig(f"best_answer_{metric}_boxplot.png", dpi=300)
    plt.close()

    # === 图2：小提琴图（Violin Plot） ===
    plt.figure(figsize=(8,5))
    sns.violinplot(x=correct_col, y=metric_col, data=df,
                   palette={"True": "green", "False": "red"},
                   inner="quartile", linewidth=1.2)
    sns.stripplot(x=correct_col, y=metric_col, data=df,
                  color="black", alpha=0.3, jitter=True)
    plt.title(f"[Violin] {metric} vs Correctness (Best Answer per Sample)")
    plt.xlabel("Correctness (True=Correct, False=Incorrect)")
    plt.ylabel(f"Mean {metric}")
    plt.tight_layout()
    plt.savefig(f"best_answer_{metric}_violin.png", dpi=300)
    plt.close()

    # === 图3：KDE密度图 ===
    plt.figure(figsize=(8,5))
    sns.kdeplot(
        df[df.correct == "True"][metric_col],
        label="Correct", fill=True, color="green", alpha=0.5
    )
    sns.kdeplot(
        df[df.correct == "False"][metric_col],
        label="Incorrect", fill=True, color="red", alpha=0.5
    )
    plt.title(f"[KDE] Distribution of {metric} by Correctness")
    plt.xlabel(f"Mean {metric}")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"best_answer_{metric}_kde.png", dpi=300)
    plt.close()

    print(f"✅ 已生成 {metric} 的 Boxplot / Violin / KDE 三种图")

print("\n所有指标分析图已生成并保存在当前文件夹。")
