"""
Pass@1 evaluation main function for MATH-500 dataset (vLLM version, silent mode, batch processing)
"""

import os
import re
import sys
import json
import time
import argparse
import contextlib
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import torch

# 禁用不必要的优化
os.environ["TRITON_ALLOW_MMA"] = "0"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from core.config import Config
from utils.common import (
    extract_model_answer,
    is_correct_answer,
    clean_latex_format
)


# ==========================
# 静音上下文管理器
# ==========================
@contextlib.contextmanager
def suppress_vllm_output():
    """临时屏蔽 vLLM 的控制台输出（包括 Rich 进度条）"""
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def pass_at_1_evaluation_vllm(dataset, config, llm, tokenizer, save_results=False, batch_size=8):
    """
    使用 vLLM 批量生成实现 Pass@1 评估函数（静音 + 批处理版本）
    """
    table = []
    n_true_ans = 0
    n_samples = 0
    start = time.time()
    index = 0

    # vLLM 采样参数
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=1.0,
        max_tokens=config.max_tokens
    )

    progress_bar = tqdm(range(0, len(dataset), batch_size), desc="Processing", leave=False)

    for start_idx in progress_bar:
        batch = dataset[start_idx : start_idx + batch_size]

        # ✅ 将 Batch 转换为样本字典列表
        batch_data = [
            {k: v[i] for k, v in batch.items()}
            for i in range(len(batch['problem']))
        ]

        prompts = []
        for data in batch_data:
            question = data['problem']
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": f"Q: {question}\nLet's think step by step and the final answer within \\boxed{{}}\nA:"}],
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)

        # 🚫 批量生成，静音
        with suppress_vllm_output():
            outputs = llm.generate(prompts, sampling_params)

        for i, data in enumerate(batch_data):
            true_answer = clean_latex_format(data['answer'])
            response_text = outputs[i].outputs[0].text

            # 清理生成文本
            cleaned_text = re.sub(r'<\|eot_id\|>', '', response_text)
            cleaned_text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', cleaned_text, flags=re.DOTALL)
            cleaned_text = re.sub(r'<\|.*?\|>', '', cleaned_text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

            # 提取模型答案
            model_answer = extract_model_answer(cleaned_text)
            model_answer = clean_latex_format(model_answer)
            is_correct = is_correct_answer(model_answer, true_answer)
            if is_correct:
                n_true_ans += 1
            n_samples += 1

            # 存储结果
            table.append({
                "ID": index + 1,
                "question": data['problem'],
                "response": cleaned_text,
                "model_answer": model_answer,
                "true_answer": true_answer,
                "is_correct": is_correct
            })
            index += 1

            # 前几个样本打印调试信息
            if index <= 3:
                print(f"\n--- 样本 {index} 调试信息 ---")
                print(f"问题: {data['problem']}")
                print(f"真值答案: {true_answer}")
                print(f"模型答案: {model_answer}")
                print(f"模型生成文本: {cleaned_text}")
                print(f"是否正确: {is_correct}")
                print("--- 结束调试信息 ---\n")

        # 更新进度条
        acc_display = f"{(n_true_ans / n_samples):.4f}" if n_samples > 0 else "0.0000"
        progress_bar.set_postfix(accuracy=acc_display)

    end = time.time()
    accuracy = n_true_ans / n_samples if n_samples > 0 else 0.0
    print("########################################################################################")
    print(f"Pass@1 Accuracy: {accuracy:.4f}")
    print(f"Total samples: {n_samples}, Correct: {n_true_ans}")
    print(f"Elapsed time: {end - start:.2f} secs.")
    print("########################################################################################")

    if save_results:
        os.makedirs("./TTT_data", exist_ok=True)
        output_file = "./TTT_data/pass_at_1_results_vllm.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "results": table,
                "accuracy": accuracy,
                "total_samples": n_samples,
                "correct_samples": n_true_ans
            }, f, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_file}")

    return accuracy



def main():
    config = Config()
    parser = argparse.ArgumentParser(description="Pass@1 evaluation using vLLM (silent, batch mode)")
    parser.add_argument("--temperature", default=1, type=float)
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--save_to_json", action="store_true")
    parser.add_argument("--dataset_repo_name", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--max_tokens", default=1024, type=int)
    parser.add_argument("--subset_size", default=None, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    args = parser.parse_args()

    config.temperature = args.temperature
    config.max_tokens = args.max_tokens

    print("正在加载语言模型 (vLLM)...")
    # 🚫 用 suppress_vllm_output 静音加载模型
    with suppress_vllm_output():
        llm = LLM(model=args.model_path, dtype="half", gpu_memory_utilization=0.9)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print("模型加载完成！")

    print("正在加载数据集...")
    dataset = load_dataset(args.dataset_repo_name, "default", split="test")
    if args.subset_size:
        dataset = dataset.select(range(min(args.subset_size, len(dataset))))
    print(f"数据集加载完成，共 {len(dataset)} 个样本")

    print("开始执行Pass@1评估...")
    pass_at_1_evaluation_vllm(dataset, config, llm, tokenizer, save_results=args.save_to_json, batch_size=args.batch_size)

    print("########################################################################################")
    print("实验配置参数:")
    for arg, value in vars(args).items():
        print(f"{arg:>25} ===> {value}")
    print("########################################################################################")


if __name__ == "__main__":
    main()
