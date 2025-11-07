"""
python generate.py --model_path meta-llama/Llama-3.2-1B-Instruct --num_return_sequences 4 --temperature 0.3 --max_tokens 1024
python generate.py --model_path /root/autodl-tmp/data/models/modelscope_cache/models/lijia321/llama_first --num_return_sequences 1 --temperature 0.3 --max_tokens 1024
"""
import argparse
import json
import os
from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch

def format_prompt_math(problem: str) -> str:
    """
    构造用于数学推理的 Prompt。
    这里使用了一个通用的零样本 CoT 模板，强制要求步骤和最终答案格式。
    你可能需要根据你的模型微调时使用的具体模板进行调整（例如增加 Chat 模板的特殊标记）。
    """
    return (
        f"Problem:\n{problem}\n\n"
        f"Let's think step by step and output the final answer within \\boxed{{}}\n"
        f"Solution:\n"
    )

def main(args):
    # 1. 加载数据集
    print(f"Loading dataset: {args.dataset_name} ({args.split})")
    # 假设加载的是 MATH500，它通常有 'problem' 和 'solution' 字段
    dataset = load_dataset(args.dataset_name, split=args.split)
    
    # 如果只想测试一下，可以只取前几个
    if args.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), args.max_samples)))

    print(f"Total problems to process: {len(dataset)}")

    # 2. 准备所有 Prompts
    # vLLM 推荐一次性传入所有 prompts 以便最大化吞吐量
    prompts = [format_prompt_math(item['problem']) for item in dataset]

    # 3. 配置采样参数
    # 关键参数是 `n`，它告诉 vLLM 为每个 prompt 生成多少条不同的路径
    sampling_params = SamplingParams(
        n=args.num_return_sequences,     # 生成 N 条回答
        temperature=args.temperature,    # 温度，建议 0.6-0.8 以保证多样性
        top_p=args.top_p,               # Nucleus sampling
        max_tokens=args.max_tokens,  # 每个回答的最大长度
        # stop=["Problem:", "Solution:"] # 可选：防止模型生成完答案后继续自言自语
    )

    # 4. 初始化 vLLM 引擎
    print(f"Initializing vLLM with model: {args.model_path}")
    # tensor_parallel_size 根据你的 GPU 数量自动设置，如果只有1张卡就是1
    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=num_gpus, 
        trust_remote_code=True,
        dtype="auto" # 自动选择 bfloat16 或 float16
    )

    # 5. 执行生成
    print("Starting generation... (this might take a while depending on N and dataset size)")
    # vLLM 会自动处理 batching 和调度
    request_outputs = llm.generate(prompts, sampling_params)

    # 6. 保存结果到 JSONL
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for i, request_output in enumerate(request_outputs):
            # 获取原始数据中的 problem 和 solution
            original_item = dataset[i]
            
            # request_output.outputs 是一个包含 N 个 CompletionOutput 对象的列表
            # 我们需要提取每个对象的 .text 属性
            generated_responses = [completion.text for completion in request_output.outputs]

            record = {
                "problem": original_item['problem'],
                "solution": original_item['solution'],
                "responses": generated_responses
            }
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 vLLM 高效生成 N 条候选回答")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--model_path", type=str, required=True, help="本地模型路径或 HF 模型名称")
    parser.add_argument("--output_file", type=str, default="math500_candidates_1.jsonl")
    
    # 核心生成参数
    parser.add_argument("--num_return_sequences", type=int, default=4, help="N: 每个问题生成的回答数量")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1024)
    
    parser.add_argument("--max_samples", type=int, default=None, help="仅用于调试：限制处理的问题数量")

    args = parser.parse_args()
    main(args)