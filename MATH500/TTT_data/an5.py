import json
from collections import Counter
from tqdm import tqdm
import random

input_path = "math500_eval_qwen1.5b_instruct.jsonl"      # 输入文件路径
output_path = "Self-Consistency.jsonl"  # 输出文件路径

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
with open(output_path, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# === 输出统计信息 ===
total = len(results)
accuracy = correct_count / total if total > 0 else 0.0

print(f"✅ 已保存到 {output_path}")
print(f"📊 总样本数: {total}")
print(f"✅ 正确数: {correct_count}")
print(f"📈 正确率: {accuracy:.4f}")
print(f"⚖️ 平票次数: {tie_count}")