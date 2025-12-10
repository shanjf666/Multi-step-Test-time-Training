# import json
# import numpy as np
# from tqdm import tqdm

# input_path = "math500_eval_qwen1.5b_instruct.jsonl"   # 输入文件路径
# output_path = "best_high_self_certainty_answers.jsonl"  # 输出文件路径

# results = []

# with open(input_path, "r") as f:
#     dataset = [json.loads(line.strip()) for line in f]

# for sample_idx, sample in enumerate(tqdm(dataset, desc="Selecting highest-entropy answers")):
#     # 尽量从样本里取 question 字段（兼容不同命名）
#     question = sample.get("question") or sample.get("problem") or sample.get("prompt") or ""
#     answers = sample.get("answers", [])

#     best_entropy = -float("inf")
#     best_answer_obj = None
#     best_answer_index = None
#     best_avg_entropy = None

#     for idx, ans in enumerate(answers):
#         steps = ans.get("steps", [])
#         if not steps:
#             continue

#         entropies = [s.get("entropy") for s in steps if s.get("entropy") is not None]
#         if not entropies:
#             continue
#         avg_entropy = float(np.mean(entropies))

#         if avg_entropy > best_entropy:
#             best_entropy = avg_entropy
#             best_answer_obj = ans
#             best_answer_index = idx
#             best_avg_entropy = avg_entropy

#     if best_answer_obj is not None:
#         # 优先取完整文本字段 full_text；没有则取 final_answer；再没有则取整个 answer 对象的 str
#         answer_text = best_answer_obj.get("full_text") or best_answer_obj.get("final_answer") or json.dumps(best_answer_obj, ensure_ascii=False)
#         # 封装 path 信息：如果 answer 中有 answer_id 字段，用它；否则使用索引
#         answer_id = best_answer_obj.get("answer_id")
#         if answer_id is not None:
#             answer_path = f"answers[answer_id={answer_id}].full_text"
#         else:
#             answer_path = f"answers[{best_answer_index}].full_text"

#         results.append({
#             "question": question,
#             "answer": answer_text,
#         })

# # 写出 JSONL
# with open(output_path, "w") as f:
#     for r in results:
#         f.write(json.dumps(r, ensure_ascii=False) + "\n")

# print(f"✅ 已保存到 {output_path}，共 {len(results)} 条记录。")

import json
import numpy as np
from tqdm import tqdm

input_path = "math500_eval_qwen1.5b_instruct.jsonl"   # 输入文件路径
output_path = "best_self_certainty_answers.jsonl"  # 输出文件路径

results = []

# === 读取 JSONL ===
with open(input_path, "r") as f:
    dataset = [json.loads(line.strip()) for line in f]

# === 逐样本处理 ===
for sample in tqdm(dataset, desc="Selecting highest self-certainty answers"):
    question = sample.get("question") or sample.get("problem") or ""
    answers = sample.get("answers", [])

    best_score = -float("inf")
    best_answer_text = None

    for ans in answers:
        steps = ans.get("steps", [])
        if not steps:
            continue

        # 计算该回答的平均 self_certainty
        certainties = [s.get("self_certainty") for s in steps if s.get("self_certainty") is not None]
        if not certainties:
            continue
        avg_certainty = np.mean(certainties)

        # 选平均 self-certainty 最大的回答
        if avg_certainty > best_score:
            best_score = avg_certainty
            best_answer_text = ans.get("full_text") or ans.get("final_answer")

    if best_answer_text:
        results.append({
            "question": question,
            "answer": best_answer_text
        })

# === 写出新 JSONL ===
with open(output_path, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ 已保存到 {output_path}，共 {len(results)} 条记录。")

