"""
python analyze.py --model_path meta-llama/Llama-3.2-1B-Instruct --temperature 0.1 --subset_size 3
"""
import os
import json 
import re
import torch
import argparse
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
# === 导入项目内模块 ===
from core.config import Config  
from utils.common import (
    parse_structured_steps,
    extract_model_answer,
    is_correct_answer,
    generate_with_transformers,
    calculate_step_self_evaluation,
    compute_CoE_C
)

def generate_evaluation_jsonl(dataset, config, model, tokenizer, device, N=4):
    results = []

    for i, data in enumerate(tqdm(dataset, desc="Generating JSONL samples")):
        question = data["problem"]
        true_answer = data["answer"]
        sample_id = f"sample_{i}"

        # 构建 prompt
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Q: {question}\nLet's think step by step and the final answer within \\boxed{{}}\nA:"}],
            tokenize=False, add_generation_prompt=True
        )

        # 生成 N 个候选
        try:
            candidates = generate_with_transformers(
                model, tokenizer, prompt, device,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                num_return_sequences=N,
                output_hidden_states=True
            )
        except Exception as e:
            print(f"Error generating candidates for sample {i}: {e}")
            continue

        answers = []

        for j, candidate in enumerate(candidates):
            response_text = tokenizer.decode(candidate["tokens"], skip_special_tokens=False)
            cleaned_text = re.sub(r'<\|eot_id\|>', '', response_text)
            cleaned_text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', cleaned_text, flags=re.DOTALL)
            cleaned_text = re.sub(r'<\|.*?\|>', '', cleaned_text)
            cleaned_text = cleaned_text.replace("<|end_of_text|>", "")
            cleaned_text = cleaned_text.replace("<|end_of_sentence|>", "")
            cleaned_text = cleaned_text.replace("</s>", "")
            cleaned_text = re.sub(r'<｜end of sentence｜>', '', cleaned_text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            cleaned_text = re.sub(r'<|endoftext|>', '', cleaned_text)
            steps = parse_structured_steps(cleaned_text)
            final_answer = extract_model_answer(cleaned_text)

            # === Self-Eval ===
            self_eval_scores = calculate_step_self_evaluation(
                model, tokenizer, steps, question, device
            )

            # === Logits 处理 ===
            logits = candidate["logits"]          # shape: [seq_len, vocab_size] 或类似
            response_ids = candidate["tokens"]    # 只包括生成部分
            prompt_len = len(candidate.get("prompt_ids", []))

            # ⚠️ 只取生成部分的 logits（对齐策略）
            try:
                if logits.shape[0] > prompt_len:
                    # 这里选取 prompt_len 到 prompt_len+len(response_ids) 的 logits 对应生成 token 的概率分布
                    start_idx = prompt_len
                    end_idx = prompt_len + len(response_ids)
                    response_logits = logits[start_idx:end_idx, :]
                else:
                    # 回退选取最后 len(response_ids) 个 logit
                    response_logits = logits[-len(response_ids):, :]
            except Exception:
                print(f"Warning: invalid logits shape at sample {i}, candidate {j}, skipping candidate.")
                continue

            if response_logits.shape[0] != len(response_ids):
                # 形状不匹配 -> 跳过该 candidate（更鲁棒）
                print(f"Warning: logits length mismatch at sample {i}, candidate {j} (logits_len={response_logits.shape[0]} vs resp_len={len(response_ids)})")
                continue

            # === 转 log_probs 和 entropy ===
            # response_logits: [T, V]
            with torch.no_grad():
                log_probs_matrix = torch.log_softmax(response_logits, dim=-1)  # [T, V]
                probs_matrix = torch.softmax(response_logits, dim=-1)
                entropy = -(probs_matrix * log_probs_matrix).sum(dim=-1)      # [T]
                # 计算 max_entropy（标量），保护类型
                if entropy.numel() > 0:
                    max_entropy_val = float(torch.max(entropy).item())
                else:
                    max_entropy_val = 0.0

            # === 步骤指标计算 ===
            step_details = []
            token_index = 0

            for k, step_text in enumerate(steps):
                if not step_text.strip():
                    continue

                step_tokens = tokenizer.encode(step_text, add_special_tokens=False)
                step_length = min(len(step_tokens), len(response_ids) - token_index)
                if step_length <= 0:
                    continue

                # === Step Probability / Certainty / Entropy ===
                step_cumulative_logprob = 0.0
                token_confidences = []
                step_entropies = []

                # iterate tokens of this step with safety checks
                for t in range(step_length):
                    abs_idx = token_index + t
                    if abs_idx < 0 or abs_idx >= len(response_ids) or abs_idx >= log_probs_matrix.shape[0]:
                        continue

                    actual_token_id = int(response_ids[abs_idx].item())
                    token_log_probs = log_probs_matrix[abs_idx]  # shape [V]

                    # 防止 vocab 越界
                    if actual_token_id < 0 or actual_token_id >= token_log_probs.shape[0]:
                        continue

                    # 累加 log-prob（用于 step_probability）
                    step_cumulative_logprob += float(token_log_probs[actual_token_id].item())

                    # === Self-certainty（你要求的实现）===
                    # "均值负对数概率" 的近似：先对 token 的分布求均值 log，然后取负号
                    token_confidence = -float(torch.mean(token_log_probs).item())
                    token_confidences.append(token_confidence)

                    # === 收集该 token 的熵（用于 step 熵）===
                    if abs_idx < entropy.shape[0]:
                        step_entropies.append(float(entropy[abs_idx].item()))

                # 归一化 step_cumulative_logprob（按有效 token 数）
                valid_token_count = len(token_confidences)
                if valid_token_count > 0:
                    step_cumulative_logprob /= valid_token_count
                else:
                    # 如果没有有效 token，则跳过该 step 的指标计算（或设为默认）
                    step_cumulative_logprob = float("-inf")

                # step probability (平均 log prob 的 exp)
                step_probability = float(np.exp(step_cumulative_logprob)) if valid_token_count > 0 else 0.0

                # Self-certainty: 平均 token_confidence（你提供的公式）
                self_certainty = float(np.mean(token_confidences)) if token_confidences else 0.0

                # === 熵 -> 归一化后映射为置信度（熵越低置信度越高） ===
                if step_entropies:
                    avg_entropy = np.mean(step_entropies)
                    # 保护 max_entropy_val 为 0 的情况
                    if max_entropy_val > 0:
                        normalized_entropy = avg_entropy / max_entropy_val
                        step_confidence_entropy = 1.0 - normalized_entropy
                        # clamp 到 [0,1]
                        step_confidence_entropy = max(0.0, min(1.0, step_confidence_entropy))
                    else:
                        # 若最大熵为0，说明分布极端确定，置信度为1
                        step_confidence_entropy = 1.0
                else:
                    step_confidence_entropy = 0.0

                # === CoE-C ===
                coe_c_score = compute_CoE_C(token_index, step_length, candidate.get("hidden_states", [])) \
                    if candidate.get("hidden_states", []) else 0.0

                # === Self-Eval ===
                self_eval = float(self_eval_scores[k]) if (k < len(self_eval_scores)) else 0.0

                step_details.append({
                    "step_id": k + 1,
                    "step_text": step_text,
                    "step_probability": step_probability,
                    "self_certainty": self_certainty,
                    "entropy": step_confidence_entropy,
                    "self_eval": self_eval,
                    "coe_c": coe_c_score
                })

                token_index += step_length

            # === 答案正确性 ===
            correct = is_correct_answer(final_answer, true_answer)

            answers.append({
                "answer_id": j,
                "full_text": cleaned_text,
                "steps": step_details,
                "final_answer": final_answer,
                "correct": correct
            })

        results.append({
            "id": sample_id,
            "question": question,
            "true_answer": true_answer,
            "answers": answers,
        })

    return results
    
def main():
    # 创建配置对象
    config = Config()

    # 命令行参数
    parser = argparse.ArgumentParser(description="基于步骤置信度的评估方法（Transformers）")

    # 生成相关
    parser.add_argument("--n_repetitive_sampling", default=4, type=int, help="为每个问题生成的解决方案数量")
    parser.add_argument("--temperature", default=0.7, type=float, help="生成时的温度参数，控制随机性")
    parser.add_argument("--top_p", default=1.0, type=float, help="Top-p采样参数")
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-Math-1.5B-Instruct", help="基础模型路径")
    parser.add_argument("--save_to_json", default=True, action="store_true", help="是否将结果保存到JSONL文件")
    parser.add_argument("--dataset_repo_name", default="HuggingFaceH4/MATH-500", help="数据集仓库名称")
    parser.add_argument("--max_tokens", default=1024, type=int, help="最大生成标记数")
    parser.add_argument("--subset_size", default=None, type=int, help="测试子集大小（用于快速测试）")
    parser.add_argument("--lambda_weight", default=0.5, type=float,
                        help="置信度计算中的lambda权重参数，用于平衡token置信度和步骤概率")
    parser.add_argument("--output_dir", default="./TTT_data", type=str, help="JSONL 输出目录")
    parser.add_argument("--device", default=None, type=str, help="强制设备，例如 'cpu' 或 'cuda'（默认自动检测）")

    args = parser.parse_args()

    # 将解析的参数赋值给配置对象
    config.n = int(args.n_repetitive_sampling)
    config.temperature = float(args.temperature)
    config.top_p = float(args.top_p)
    config.max_tokens = int(args.max_tokens)
    config.lambda_weight = float(args.lambda_weight)

    # 设备选择
    if args.device:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # 加载基础语言模型（根据是否有 GPU 设定 dtype）
    print("正在加载语言模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side='left',  # <--- 关键修复
        trust_remote_code=True
    )

    # 添加 padding token 如果不存在
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    print("语言模型加载完成!")

    # 加载数据集
    print(f"正在加载数据集...")
    dataset = load_dataset(args.dataset_repo_name, 'default', split="test")

    # 测试子集
    if args.subset_size:
        take = min(args.subset_size, len(dataset))
        dataset = dataset.select(range(take))

    print(f"数据集加载完成，共 {len(dataset)} 个样本")

    # 调用生成函数（generate_evaluation_jsonl 在你的项目中实现）
    print("开始生成评估 JSONL 数据（逐条 sample）...")
    results = generate_evaluation_jsonl(
        dataset=dataset,
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        N=config.n
    )

    # 保存 JSONL
    if args.save_to_json:
        os.makedirs(args.output_dir, exist_ok=True)
        outpath = os.path.join(args.output_dir, f"math500_eval_llama1b.jsonl")
        with open(outpath, "w", encoding="utf-8") as fout:
            for rec in results:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n✅ 已保存 {len(results)} 条样本到: {outpath}")

    else:
        print("save_to_json 为 False，未保存到文件。")
        
        # 打印所有参数及其值
    print("########################################################################################")
    print("实验配置参数:")
    print("########################################################################################")
    for arg, value in vars(args).items():
        print(f"{arg:>25} ===> {value}")
    print("########################################################################################")


if __name__ == "__main__":
    main()

