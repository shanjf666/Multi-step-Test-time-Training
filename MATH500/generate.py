"""
generate.py
python generate.py --model_path meta-llama/Llama-3.2-1B-Instruct --num_return_sequences 1 --temperature 0.1 --max_tokens 1024 --output_file math500_candidates.jsonl
python generate.py --model_path meta-llama/Llama-3.2-1B-Instruct --num_return_sequences 4 --temperature 0.1 --max_tokens 1024 --output_file math500_candidates_0.jsonl
python generate.py --model_path /root/autodl-tmp/data/models/modelscope_cache/models/lijia321/llama_first --num_return_sequences 4 --temperature 0.1 --max_tokens 1024 --output_file math500_candidates_1.jsonl
python generate.py --model_path /root/autodl-tmp/data/models/modelscope_cache/models/lijia321/llama_second --num_return_sequences 4 --temperature 0.1 --max_tokens 1024 --output_file math500_candidates_2.jsonl
"""
import argparse
import json
import os
from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer

def format_prompt_math(problem: str, tokenizer) -> str:
    """
    使用 tokenizer.apply_chat_template 构造 chat-style prompt。
    这里把问题放到 user role，并在末尾加上生成提示 "A:"（便于模型以 assistant 身份输出）。
    """
    # 你要求的模板形式（把 problem 变量改为 question 的形式也可以）
    user_content = f"Q: {problem}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"
    # 使用 apply_chat_template 生成 chat-style prompt，保持 tokenize=False 以传给 vLLM 原始字符串
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True
    )

def main(args):
    # 1. 加载 tokenizer（用于生成 chat 模板）
    print(f"Loading tokenizer for model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    # 2. 加载数据集
    print(f"Loading dataset: {args.dataset_name} ({args.split})")
    dataset = load_dataset(args.dataset_name, split=args.split)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))
    print(f"Total problems to process: {len(dataset)}")

    # 3. 准备所有 Prompts（使用 chat 模板）
    prompts = [format_prompt_math(item['problem'], tokenizer) for item in dataset]

    # 4. 配置采样参数（建议固定 seed，加入 stop marker）
    sampling_params = SamplingParams(
        n=args.num_return_sequences,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop=["<|eot_id|>", "</s>", "Q:"],  # 根据模型仓库的结束标志调整
    )

    # 5. 初始化 vLLM 引擎
    print(f"Initializing vLLM with model: {args.model_path}")
    num_gpus = torch.cuda.device_count() or 1
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        dtype="auto"
    )

    # 6. 执行生成
    print("Starting generation...")
    request_outputs = llm.generate(prompts, sampling_params)

    # 7. 保存结果到 JSONL（使用 dataset 的 answer 字段作为真实答案）
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for i, request_output in enumerate(request_outputs):
            original_item = dataset[i]
            generated_responses = [completion.text for completion in request_output.outputs]
            record = {
                "problem": original_item['problem'],
                "answer": original_item.get('answer', original_item.get('solution', None)),
                "responses": generated_responses
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 vLLM 高效生成 N 条候选回答（chat-template 版）")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_path", type=str, required=True, help="本地模型路径或 HF 模型名称")
    parser.add_argument("--output_file", type=str, default="math500_candidates.jsonl")

    # 核心生成参数
    parser.add_argument("--num_return_sequences", type=int, default=4, help="N: 每个问题生成的回答数量")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于可复现采样）")

    parser.add_argument("--max_samples", type=int, default=None, help="仅用于调试：限制处理的问题数量")

    args = parser.parse_args()
    main(args)
