import json
import argparse
import numpy as np
import os
import gc
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset
from math import exp, log
from math_utils import extract_answer, is_correct, parse_steps

def calculate_vllm_metrics(request_output, top_k):
    """
    直接从 vLLM 的 logprobs 输出计算 Top-K Entropy
    """
    token_entropies = []
    token_certainties = [] 
    
    if not request_output.logprobs:
        return 0, 0, 0, 0, 0, 0
        
    for step_logprobs in request_output.logprobs:
        if not step_logprobs: continue
        vals = list(step_logprobs.values())
        token_certainties.append(vals[0].logprob)
        
        probs = [exp(lp.logprob) for lp in vals]
        sum_p = sum(probs)
        
        if sum_p > 0:
            norm_probs = [p / sum_p for p in probs]
            ent = -sum(p * log(p + 1e-12) for p in norm_probs)
            token_entropies.append(ent)
        else:
            token_entropies.append(0.0)
            
    if not token_entropies:
        return 0, 0, 0, 0, 0, 0
        
    avg_ent = np.mean(token_entropies)
    std_ent = np.std(token_entropies)
    avg_cert = np.mean(token_certainties)
    
    return avg_ent, std_ent, avg_cert, token_entropies, token_certainties

def main(args):
    # 1. 加载数据
    print(f"Loading dataset: {args.dataset}...")
    prompt_template = "Question: {q}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\nAnswer:"
    
    prompts, answers = [], []
    if args.dataset.lower() == 'gsm8k':
        ds = load_dataset("gsm8k", "main", split=args.split)
        prompts = [prompt_template.format(q=item['question']) for item in ds]
        answers = [item['answer'] for item in ds]
    elif 'math' in args.dataset.lower() or 'aime' in args.dataset.lower():
        ds = load_dataset(args.dataset, split=args.split)
        prompts = [f"Problem: {item['problem']}\nSolution:" for item in ds]
        answers = [item['solution'] for item in ds]
    elif args.dataset.startswith("AI-MO") or "aimo" in args.dataset.lower():  # ← 添加这一段
        ds = load_dataset(args.dataset, split=args.split)
        prompts = [f"Problem: {item['problem']}\nSolution:" for item in ds]
        answers = [item['answer'] for item in ds]
    else:
        with open(args.dataset, 'r') as f:
            raw = [json.loads(line) for line in f]
        prompts = [prompt_template.format(q=item.get('problem', item.get('question'))) for item in raw]
        answers = [item.get('answer', item.get('solution')) for item in raw]

    if args.max_samples:
        prompts = prompts[:args.max_samples]
        answers = answers[:args.max_samples]

    print(f"Total problems: {len(prompts)}")

    # 2. 初始化 vLLM (关键配置优化)
    print(f"Initializing vLLM: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.gpu_count,
        trust_remote_code=True,
        gpu_memory_utilization=0.85, # [关键] 稍微降低显存占用，防止碎片化卡死
        max_model_len=4096,          # [关键] 限制最大长度，GSM8K 不需要 32k 窗口，节省显存
        swap_space=16                # [关键] 增加 CPU Swap 空间防止 OOM
    )
    
    sampling_params = SamplingParams(
        n=args.n_samples,
        temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
        logprobs=20 
    )

    # 3. [关键修改] 分块处理 (Chunk Processing)
    # 每次处理 chunk_size 个问题 (即 chunk_size * n_samples 个推理请求)
    chunk_size = 20 # 每次处理 20 个问题 (20 * 64 = 1280 个序列)，这个负载很安全
    
    # 如果文件已存在，先清空或备份，这里选择覆盖模式
    if os.path.exists(args.output_file):
        print(f"Warning: Overwriting {args.output_file}")
        open(args.output_file, 'w').close() 

    print(f"Starting chunked generation (Chunk Size: {chunk_size})...")
    
    total_chunks = (len(prompts) + chunk_size - 1) // chunk_size
    
    for chunk_idx in tqdm(range(total_chunks), desc="Processing Chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(prompts))
        
        batch_prompts = prompts[start_idx:end_idx]
        batch_answers = answers[start_idx:end_idx]
        
        # --- 生成当前块 ---
        # use_tqdm=False 防止进度条刷屏
        outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
        
        # --- 处理并写入 ---
        results_to_write = []
        for i, output in enumerate(outputs):
            problem = batch_prompts[i]
            gt = batch_answers[i]
            response_objs = []
            
            for sample in output.outputs:
                text = sample.text
                if sample.logprobs:
                    # 立即计算指标并丢弃原始 logprobs 对象以释放内存
                    avg_ent, std_ent, avg_cert, token_ents, _ = calculate_vllm_metrics(sample, 20)
                    
                    steps_text = parse_steps(text)
                    
                    # 粗略计算 Step Std
                    step_count = len(steps_text)
                    if step_count > 1 and len(token_ents) > step_count:
                        chunk_len = len(token_ents) // step_count
                        step_means = [np.mean(token_ents[j:j+chunk_len]) for j in range(0, len(token_ents), chunk_len)]
                        step_std = np.std(step_means)
                    else:
                        step_std = 0.0
                    
                    # 重新计算 max_step_ent (近似)
                    if step_count > 0:
                        chunk_len = max(1, len(token_ents) // step_count)
                        step_means = [np.mean(token_ents[j:j+chunk_len]) for j in range(0, len(token_ents), chunk_len)]
                        max_step_ent = max(step_means) if step_means else avg_ent
                    else:
                        max_step_ent = avg_ent

                    pred = extract_answer(text)
                    
                    response_objs.append({
                        "text": text,
                        "extracted_answer": pred,
                        "token_count": len(token_ents),
                        "std_topk_entropy": std_ent,
                        "avg_topk_entropy": avg_ent,
                        "avg_certainty": avg_cert,
                        "step_std_entropy": step_std,
                        "max_step_entropy": max_step_ent, 
                        "steps_count": len(steps_text)
                    })
                else:
                    response_objs.append({"text": text, "error": "No logprobs"})

            results_to_write.append({
                "problem": problem,
                "answer": gt,
                "responses": response_objs
            })
            
        # --- 增量写入文件 (Append Mode) ---
        with open(args.output_file, 'a', encoding='utf-8') as f:
            for item in results_to_write:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # --- [关键] 手动清理内存 ---
        del outputs
        del results_to_write
        gc.collect() 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_file", type=str, default="rollouts.jsonl")
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--gpu_count", type=int, default=1)
    
    args = parser.parse_args()
    main(args)