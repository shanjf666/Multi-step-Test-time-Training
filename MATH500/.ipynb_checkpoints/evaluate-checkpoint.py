"""
evaluate.py
python evaluate.py --input_file math500_candidates.jsonl --output_file bo4_certainty0.jsonl --model_dir meta-llama/Llama-3.2-1B-Instruct
python evaluate.py --input_file math500_candidates_0.jsonl --output_file bo4_certainty.jsonl --model_dir meta-llama/Llama-3.2-1B-Instruct
python evaluate.py --input_file math500_candidates_1.jsonl --output_file bo4_certainty_first.jsonl --model_dir /root/autodl-tmp/data/models/modelscope_cache/models/lijia321/llama_first
python evaluate.py --input_file math500_candidates_2.jsonl --output_file bo4_certainty_second.jsonl --model_dir /root/autodl-tmp/data/models/modelscope_cache/models/lijia321/llama_second
"""
import argparse
import json
import os
import re
import tqdm
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import numpy as np

def clean_latex_format(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    cleaned_text = text.strip()

    cleaned_text = cleaned_text.replace("$", "").replace("\\$", "$")

    latex_whitespaces = {
        r'\\,': '', r'\\:': '', r'\\;': '', r'\\!': '',
        r'\\enspace': '', r'\\quad': '', r'\\qquad': '',
        r'\\hspace{[^}]*}': '', r'\\vspace{[^}]*}': '',
        r'\\phantom{[^}]*}': '', r'\\hfill': '', r'\\space': '',
        r'\\ ': '', r'\\mspace{[^}]*}': '', r'\\kern{[^}]*}': '',
    }
    for pattern, replacement in latex_whitespaces.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text)

    useless_cmds = [
        r'\\left', r'\\right', r'\\big', r'\\Big', r'\\bigg', r'\\Bigg',
        r'\\text\s*', r'\\mathrm', r'\\displaystyle', r'\\rm', r'\\it',
        r'\\bf', r'\\cal', r'\\scriptstyle', r'\\scriptscriptstyle'
    ]
    for cmd in useless_cmds:
        cleaned_text = re.sub(cmd, '', cleaned_text)

    cleaned_text = cleaned_text.replace(r'\(', '(').replace(r'\)', ')')
    cleaned_text = cleaned_text.replace(r'\[', '[').replace(r'\]', ']')
    cleaned_text = cleaned_text.replace(r'\{', '{').replace(r'\}', '}')

    symbol_replacements = {
        r'\\pi': 'pi', r'\\theta': 'theta', r'\\sqrt': 'sqrt',
        r'\\frac{([^}]*?)}({[^}]*?})': r'\1/\2',  # \frac{a}{b} -> a/b
        r'\\times': '*', r'\\div': '/', r'\\infty': 'oo',
        r'\\alpha': 'alpha', r'\\beta': 'beta', r'\\gamma': 'gamma',
        r'\\delta': 'delta', r'\\sum': 'sum', r'\\int': 'integrate',
        r'\\cdot': '*', r'\\pm': '+-', r'\\mp': '-+'
    }
    for pattern, replacement in symbol_replacements.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text)

    np_pattern = re.compile(r'[\x00-\x1f]')
    cleaned_text = np_pattern.sub('', cleaned_text)

    cleaned_text = re.sub(r'\(\s+', '(', cleaned_text)  # (  -> (
    cleaned_text = re.sub(r'\s+\)', ')', cleaned_text)  #  ) -> )
    cleaned_text = re.sub(r'\[\s+', '[', cleaned_text)  # [  -> [
    cleaned_text = re.sub(r'\s+\]', ']', cleaned_text)  #  ] -> ]
    cleaned_text = re.sub(r'\{\s+', '{', cleaned_text)  # {  -> {
    cleaned_text = re.sub(r'\s+\}', '}', cleaned_text)  #  } -> }
    # 移除逗号前后空格
    cleaned_text = re.sub(r'\s*,\s*', ',', cleaned_text)
    # 规范化多余空格
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    if not cleaned_text:
        return ""

    return cleaned_text

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]
    return retval


def remove_boxed(s):
    if s is None:
        return None
    if "\\boxed " in s:
        return s.replace("\\boxed ", "")
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[len("\\boxed{") : -1]
    return s


def strip_string(string):
    if string is None:
        return ""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace(" ", "")
    if string == "0.5": string = "\\frac{1}{2}"
    return string


# ---------- 核心封装函数 ----------

def extract_model_answer(response_text: str) -> str:
    """
    从模型输出中提取最终答案（去掉 \\boxed{}）。
    如果没有找到 \\boxed，返回 None。
    """
    boxed_part = last_boxed_only_string(response_text)
    if boxed_part is None:
        return None
    answer = remove_boxed(boxed_part)
    return answer.strip()


def is_correct_answer(predicted: str, true_answer: str) -> bool:
    """
    判断预测答案与标准答案是否等价（忽略LaTeX格式差异）。
    """
    if predicted is None or true_answer is None:
        return False
    return strip_string(predicted) == strip_string(true_answer)


def parse_structured_steps(response_text: str, min_len: int = 5) -> List[Tuple[int, int, str]]:
    steps = []
    step_pattern = re.compile(r'(?:##\s*)?Step\s*(?:\d+)\s*[:.]?\s*(.*?)(?=(?:##\s*)?Step\s*\d+\s*[:.]?|$)', re.DOTALL | re.IGNORECASE)
    matches = list(step_pattern.finditer(response_text))
    valid_matches = []
    for m in matches:
        content = m.group(1).strip()
        if len(content) >= min_len:
            valid_matches.append((m.start(1), m.end(1), content))
    if valid_matches:
        return valid_matches

    para_pattern = re.compile(r'\S.*?(?:\n\s*\n|$)', re.DOTALL)
    paragraphs = []
    for m in para_pattern.finditer(response_text):
        content = m.group(0).strip()
        if len(content) >= min_len and not content.startswith("<|"):
            paragraphs.append((m.start(), m.end(), content))
    if len(paragraphs) >= 2:
        return paragraphs

    sentence_pattern = re.compile(r'[^\.!\?]+[\.!\?]+', re.DOTALL)
    sentences = []
    for m in sentence_pattern.finditer(response_text):
        content = m.group(0).strip()
        if len(content) >= min_len:
            sentences.append((m.start(), m.end(), content))
    if len(sentences) >= 2:
        return sentences

    return [(0, len(response_text), response_text)]

def compute_token_certainty(logits: torch.Tensor, input_ids: torch.Tensor, V: int) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    logprob_sum = log_probs.sum(dim=-1)
    V_tensor = torch.tensor(V, dtype=logits.dtype, device=logits.device)
    token_scores = -1 / V * logprob_sum - torch.log(V_tensor)
    prefix = torch.zeros((token_scores.size(0), 1), dtype=token_scores.dtype, device=token_scores.device)
    token_scores = torch.cat([prefix, token_scores], dim=1)
    return token_scores

def compute_token_neg_entropy(logits: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    probs = F.softmax(shift_logits, dim=-1)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum(dim=-1)
    prefix = torch.zeros((entropy.size(0), 1), dtype=entropy.dtype, device=entropy.device)
    neg_entropy = torch.cat([prefix, -entropy], dim=1)
    return neg_entropy

@torch.no_grad()
def process_math_data(args):
    # --- 加载数据 ---
    ext = os.path.splitext(args.input_file)[1].lower()
    if ext == '.jsonl':
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    elif ext == '.json':
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        raise ValueError("Unsupported file format")

    # --- 加载模型 ---
    print(f"Loading model from {args.model_dir}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    llm = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm.eval()

    alpha = args.alpha

    # ------------ 新文件名：基于 output_file 生成 ------------
    if args.output_file:
        score_file = os.path.splitext(args.output_file)[0] + ".jsonl"
    else:
        score_file = os.path.splitext(args.input_file)[0] + "_scores.jsonl"

    f_score = open(score_file, 'w', encoding='utf-8')
    # ----------------------------------------------------------

    processed_count = 0
    total_correct = 0
    actual_processed_count = 0

    # 保存用于阈值分析
    stats_records = []

    # =============================
    # 主循环
    # =============================
    pbar = tqdm.tqdm(enumerate(data), total=len(data), desc="Processing")
    try:
        for i, item in pbar:
            problem_text = item.get("problem", "") or item.get("question", "")
            candidate_responses = item.get("responses", [])
            true_solution = item.get("solution", "") or item.get("answer", "")

            if not candidate_responses:
                continue

            prompt_len_char = len(problem_text)
            all_token_scores_cpu, all_neg_entropy_cpu, all_offsets_cpu, all_input_ids_cpu = [], [], [], []

            # --- 处理候选回答 ---
            num_responses = len(candidate_responses)
            for batch_start in range(0, num_responses, args.forward_batch_size):
                batch_end = min(batch_start + args.forward_batch_size, num_responses)
                batch_responses = candidate_responses[batch_start:batch_end]
                full_texts = [problem_text + r for r in batch_responses]

                inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True).to(device)
                with torch.amp.autocast(device_type='cuda', dtype=dtype):
                    outputs = llm(inputs.input_ids, attention_mask=inputs.attention_mask)
                    batch_certainty = compute_token_certainty(outputs.logits, inputs.input_ids, llm.config.vocab_size)
                    batch_neg_entropy = compute_token_neg_entropy(outputs.logits)

                all_token_scores_cpu.extend([s.cpu() for s in batch_certainty])
                all_neg_entropy_cpu.extend([s.cpu() for s in batch_neg_entropy])
                all_offsets_cpu.extend([o.cpu() for o in inputs.offset_mapping])
                all_input_ids_cpu.extend([ids.cpu() for ids in inputs.input_ids])
                del outputs, batch_certainty, batch_neg_entropy, inputs
                torch.cuda.empty_cache()

            candidates_res = []
            best_score = -float('inf')
            best_res_data = None

            for idx in range(num_responses):
                response_text = candidate_responses[idx]
                token_scores = all_token_scores_cpu[idx]
                neg_entropy = all_neg_entropy_cpu[idx]
                offsets = all_offsets_cpu[idx]
                input_ids = all_input_ids_cpu[idx]

                # 找到回答起始token位置
                res_start_token_idx = 0
                for t_i in range(len(input_ids)):
                    if input_ids[t_i] == tokenizer.pad_token_id:
                        continue
                    if offsets[t_i][0] >= prompt_len_char:
                        res_start_token_idx = t_i
                        break

                # 步骤划分
                steps = parse_structured_steps(response_text)
                step_scores_list = []

                for s_i, (c_start, c_end, s_text) in enumerate(steps):
                    abs_start, abs_end = prompt_len_char + c_start, prompt_len_char + c_end
                    step_token_mask = (offsets[:, 0] < abs_end) & (offsets[:, 1] > abs_start)
                    step_token_mask[:res_start_token_idx] = False
                    if step_token_mask.any():
                        avg_certainty = token_scores[step_token_mask].mean().item()
                        avg_neg_entropy = neg_entropy[step_token_mask].mean().item()
                        step_score = alpha * avg_certainty + (1 - alpha) * avg_neg_entropy
                        step_scores_list.append(step_score)

                final_score = np.mean(step_scores_list) if step_scores_list else -999.0
                is_correct = is_correct_answer(clean_latex_format(extract_model_answer(response_text)),
                                               clean_latex_format(true_solution))

                res_data = {
                    "idx": idx,
                    "score": final_score,
                    "correct": is_correct,
                    "text": response_text
                }
                candidates_res.append(res_data)

                if final_score > best_score:
                    best_score = final_score
                    best_res_data = res_data

            # --- Sigmoid 归一化 ---
            scores = np.array([c["score"] for c in candidates_res], dtype=float)
            valid_mask = scores > -998.0
            if valid_mask.any():
                valid_scores = scores[valid_mask]
                mean_val = valid_scores.mean()
                std_val = valid_scores.std() if valid_scores.std() > 1e-8 else 1.0
                k = 3.0
                norm_scores = 1 / (1 + np.exp(-k * (scores - mean_val) / std_val))
                for k_i, c in enumerate(candidates_res):
                    c["score_norm"] = float(norm_scores[k_i])
            else:
                for c in candidates_res:
                    c["score_norm"] = 0.0

            # 找到最佳回答（带归一化得分）
            if best_res_data:
                actual_processed_count += 1
                if best_res_data["correct"]:
                    total_correct += 1

                best_res_data = next((r for r in candidates_res if r["idx"] == best_res_data["idx"]), best_res_data)
                best_norm = best_res_data.get("score_norm", 0.0)

                stats_records.append({
                    "idx": i,
                    "score_norm": float(best_norm),
                    "correct": best_res_data["correct"]
                })

                # ----------- 写入新的 score 文件（唯一输出） -----------
                f_score.write(json.dumps({
                    "idx": i,
                    "question": problem_text,
                    "best_answer": best_res_data["text"],
                    "score_norm": float(best_norm),
                    "correct": best_res_data["correct"]
                }, ensure_ascii=False) + "\n")
                # -------------------------------------------------------

            if actual_processed_count > 0 and actual_processed_count % 5 == 0:
                pbar.set_postfix({"Acc": f"{total_correct / actual_processed_count:.2%}"})

    finally:
        f_score.close()
        print(f"\nScores saved to {score_file}")

    # =============================
    # 阈值分析（调试信息）
    # =============================
    thresholds = np.arange(0.5, 1.01, 0.01)
    scores = np.array([r["score_norm"] for r in stats_records])
    corrects = np.array([r["correct"] for r in stats_records])

    print("\n" + "=" * 40)
    print("阈值分析（基于归一化置信度）")
    print("=" * 40)
    for t in thresholds:
        mask = scores <= t
        n = mask.sum()
        acc = corrects[mask].mean() if n > 0 else 0.0
        print(f"阈值 {t:.2f} | 样本数 = {n:4d} | 准确率 = {acc:.2%}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--forward_batch_size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.4, help="步骤得分加权系数 alpha (self-certainty权重)")
    args = parser.parse_args()
    process_math_data(args)

