"""
python test.py --input_file math500_candidates.jsonl --output_file bo4_certainty_llama_1.jsonl --model_dir meta-llama/Llama-3.2-1B-Instruct
"""
import argparse
import json
import os
import re
import tqdm
from typing import List, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from utils.common import (
    extract_model_answer, 
    is_correct_answer,
    clean_latex_format
)

def parse_structured_steps(response_text: str, min_len: int = 5) -> List[Tuple[int, int, str]]:
    """
    解析结构化步骤（优先匹配 "## Step N:" 块），否则回退为段落/行。
    返回: [(start_char_idx, end_char_idx, step_content), ...]
    注意：为了保持 Offsets 准确，我们尽量在原始 response_text 上进行匹配，
    而不是在 heavily cleaned_text 上匹配。
    """
    steps = []
    
    # --- 策略 1: 优先匹配 "## Step N:" ---
    # 使用你的正则，但改用 finditer 来获取位置。
    # 我们稍微放宽正则，允许 Step 前面有换行符或其他杂音，不依赖彻底的 clean
    step_pattern = re.compile(r'(?:##\s*)?Step\s*(?:\d+)\s*[:.]?\s*(.*?)(?=(?:##\s*)?Step\s*\d+\s*[:.]?|$)', re.DOTALL | re.IGNORECASE)
    
    matches = list(step_pattern.finditer(response_text))
    # 过滤掉太短的或者看起来是误匹配的（比如只匹配到了 prompt 里的 instruction）
    valid_matches = []
    for m in matches:
        # group(1) 是步骤的具体内容
        content = m.group(1).strip()
        if len(content) >= min_len:
            valid_matches.append((m.start(1), m.end(1), content))
            
    if valid_matches:
        return valid_matches

    # 如果没有匹配到明确的 Step，我们需要回退策略。
    # 为了位置准确，我们尽量避免破坏性清洗，而是用正则去“挑出”我们想要的部分。

    # --- 策略 2: 按双换行符分段 (段落) ---
    # 查找所有非空段落
    para_pattern = re.compile(r'\S.*?(?:\n\s*\n|$)', re.DOTALL)
    paragraphs = []
    for m in para_pattern.finditer(response_text):
        content = m.group(0).strip()
        # 简单过滤掉可能是特殊 Token 的短行
        if len(content) >= min_len and not content.startswith("<|"):
             paragraphs.append((m.start(), m.end(), content))
             
    if len(paragraphs) >= 2:
        return paragraphs

    # --- 策略 3: 按句子粗分 (回退) ---
    # 在原始文本上按标点分句比较困难，这里用一个简化的近似方法：
    # 匹配任何以 .!? 结尾的非空字符序列
    sentence_pattern = re.compile(r'[^\.!\?]+[\.!\?]+', re.DOTALL)
    sentences = []
    for m in sentence_pattern.finditer(response_text):
        content = m.group(0).strip()
        if len(content) >= min_len:
            sentences.append((m.start(), m.end(), content))
            
    if len(sentences) >= 2:
        return sentences

    # --- 兜底: 整个文本作为一个步骤 ---
    return [(0, len(response_text), response_text)]

def compute_token_certainty(logits: torch.Tensor, input_ids: torch.Tensor, V: int) -> torch.Tensor:
    """
    计算一个Batch内每个有效位置Token的Self-Certainty分数。
    
    参数:
    logits: 模型输出的完整logits (batch, seq_len, vocab_size)
    input_ids: 输入的token ids (batch, seq_len)，用于移位对齐
    V: 词表大小
    """
    # Shift操作：第i个logits预测的是第i+1个token
    shift_logits = logits[..., :-1, :].contiguous()
    # shift_labels = input_ids[..., 1:].contiguous() # 在这个特定指标中其实不需要label，只需要logits分布

    # 计算 LogSoftmax 和 LogProb Sum (针对整个词表求和)
    # 公式: -1/V * \sum_{v \in V} \log P(v|x) - \log V
    log_probs = F.log_softmax(shift_logits, dim=-1)
    logprob_sum = log_probs.sum(dim=-1) # (batch, seq_len-1)

    V_tensor = torch.tensor(V, dtype=logits.dtype, device=logits.device)
    token_scores = -1/V * logprob_sum - torch.log(V_tensor)
    
    # 在序列最前面补一个0，保持和原始 input_ids 长度一致，方便后续通过下标索引
    # (batch, seq_len-1) -> (batch, seq_len)
    prefix = torch.zeros((token_scores.size(0), 1), dtype=token_scores.dtype, device=token_scores.device)
    token_scores = torch.cat([prefix, token_scores], dim=1)
    
    return token_scores


@torch.no_grad()
def process_math_data(args):
    # --- 加载数据 ---
    ext = os.path.splitext(args.input_file)[1].lower()
    if ext == '.jsonl':
        with open(args.input_file, 'r') as f: data = [json.loads(line) for line in f]
    elif ext == '.json':
        with open(args.input_file, 'r') as f: data = json.load(f)
    else:
        raise ValueError("Unsupported file format")

    # --- 加载模型 ---
    print(f"Loading model from {args.model_dir}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    llm = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    llm.eval()

    # --- 准备输出 ---
    output_file = args.output_file or os.path.splitext(args.input_file)[0] + "_best_scored.jsonl"
    processed_count = 0
    
    # [新增] 初始化统计计数器
    total_correct = 0
    actual_processed_count = 0

    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                processed_count += 1
                # 如果是断点续传，尝试从已处理的文件中恢复准确率统计（可选，这里简单起见只统计本次运行的）
                # try:
                #     rec = json.loads(line)
                #     if rec.get("best_response", {}).get("correct"): total_correct += 1
                #     actual_processed_count += 1
                # except: pass
                
    print(f"Resuming from index {processed_count}, Output to: {output_file}")
    f_out = open(output_file, 'a', encoding='utf-8')

    # --- 主循环 ---
    pbar = tqdm.tqdm(enumerate(data), total=len(data), desc="Processing")
    for i, item in pbar:
        if i < processed_count: continue

        problem_text = item.get("problem", "") or item.get("question", "")
        candidate_responses = item.get("responses", [])
        true_solution = clean_latex_format(item.get("solution", "") or item.get("answer", ""))

        if not candidate_responses:
            f_out.write(json.dumps({"problem_idx": i, "error": "no_responses"}, ensure_ascii=False) + "\n")
            continue

        prompt_len_char = len(problem_text)

        # --- 批量前向传播 ---
        all_token_scores_cpu = [] 
        all_offsets_cpu = []
        all_input_ids_cpu = []

        num_responses = len(candidate_responses)
        for batch_start in range(0, num_responses, args.forward_batch_size):
            batch_end = min(batch_start + args.forward_batch_size, num_responses)
            batch_responses = candidate_responses[batch_start:batch_end]
            full_texts = [problem_text + r for r in batch_responses]
            
            inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, 
                             return_offsets_mapping=True).to(device)
            
            with torch.amp.autocast(device_type='cuda', dtype=dtype):
                outputs = llm(inputs.input_ids, attention_mask=inputs.attention_mask)
            
            batch_scores = compute_token_certainty(outputs.logits, inputs.input_ids, llm.config.vocab_size)
            
            all_token_scores_cpu.extend([s.cpu() for s in batch_scores])
            all_offsets_cpu.extend([o.cpu() for o in inputs.offset_mapping])
            all_input_ids_cpu.extend([ids.cpu() for ids in inputs.input_ids])

            del outputs, batch_scores, inputs
            torch.cuda.empty_cache()

        # --- 指标计算与选择最佳回答 ---
        best_score = -float('inf')
        best_res_data = None

        for idx in range(num_responses):
            response_text = candidate_responses[idx]
            token_scores = all_token_scores_cpu[idx]
            offsets = all_offsets_cpu[idx]
            input_ids = all_input_ids_cpu[idx]

            # 1. 定位 Response 开始位置
            res_start_token_idx = 0
            for t_i in range(len(input_ids)):
                if input_ids[t_i] == tokenizer.pad_token_id: continue
                if offsets[t_i][0] >= prompt_len_char:
                    res_start_token_idx = t_i
                    break
            
            # 2. 划分步骤并计算分数
            steps = parse_structured_steps(response_text)
            step_scores_list = []
            step_details = []

            for s_i, (c_start, c_end, s_text) in enumerate(steps):
                abs_start = prompt_len_char + c_start
                abs_end = prompt_len_char + c_end
                step_token_mask = (offsets[:, 0] < abs_end) & (offsets[:, 1] > abs_start)
                step_token_mask[:res_start_token_idx] = False
                
                if step_token_mask.any():
                    avg_score = token_scores[step_token_mask].mean().item()
                    step_scores_list.append(avg_score)
                    step_details.append({"step": s_i, "score": avg_score, "text": s_text[:50]+"..."})

            # 3. 计算最终得分 (防止空列表导致 NaN)
            if step_scores_list:
                final_score = np.mean(step_scores_list)
            else:
                final_score = -999.0

            # 4. 评估答案
            pred_ans = clean_latex_format(extract_model_answer(response_text))
            is_correct = is_correct_answer(pred_ans, true_solution)

            res_data = {
                "idx": idx,
                "score": final_score,
                "pred": pred_ans,
                "correct": is_correct,
                "steps": step_details,
                # "full_text": response_text # 可选：如果不希望文件太大，可以注释掉
            }

            if final_score > best_score:
                best_score = final_score
                best_res_data = res_data

        # --- [新增] 更新准确率统计 ---
        if best_res_data:
            actual_processed_count += 1
            if best_res_data["correct"]:
                total_correct += 1
        
        # 实时在进度条显示准确率 (每10个更新一次显示，避免刷屏)
        if actual_processed_count > 0 and actual_processed_count % 5 == 0:
            current_acc = total_correct / actual_processed_count
            pbar.set_postfix({"Acc": f"{current_acc:.2%}"})

        # --- 保存结果 ---
        record = {
            "problem_idx": i,
            "problem": problem_text[:200]+"...", # 略微截断以节省空间，可视需要调整
            "best_response": best_res_data
        }
        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        f_out.flush()

    f_out.close()
    
    # --- [新增] 输出最终统计信息 ---
    print("\n" + "="*30)
    print("Processing Complete!")
    if actual_processed_count > 0:
        final_acc = total_correct / actual_processed_count
        print(f"Total Processed: {actual_processed_count}")
        print(f"Total Correct:   {total_correct}")
        print(f"Final Accuracy:  {final_acc:.2%} ({total_correct}/{actual_processed_count})")
    else:
        print("No problems were processed in this run.")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="JSONL文件，需包含 'problem', 'responses'(列表), 'solution'")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--model_dir", type=str, required=True)
    # 关键参数：控制模型前向传播时的批次大小，防止OOM
    parser.add_argument("--forward_batch_size", type=int, default=4, help="模型前向传播时的微批次大小")
    args = parser.parse_args()
    
    process_math_data(args)