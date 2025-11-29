"""
generate.py
python generate.py --model_path meta-llama/Llama-3.2-1B-Instruct --num_return_sequences 1 --temperature 0.1 --max_tokens 1024 --output_file math500_candidates.jsonl
python generate.py --model_path meta-llama/Llama-3.2-1B-Instruct --num_return_sequences 4 --temperature 0.1 --max_tokens 1024 --output_file math500_candidates_0.jsonl
python generate.py --model_path /root/autodl-tmp/data/models/modelscope_cache/models/lijia321/llama_first --num_return_sequences 4 --temperature 0.1 --max_tokens 1024 --output_file math500_candidates_1.jsonl
python generate.py --model_path /root/autodl-tmp/data/models/modelscope_cache/models/lijia321/llama_second --num_return_sequences 4 --temperature 0.1 --max_tokens 1024 --output_file math500_candidates_2.jsonl
"""
import json
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# =====================================================
# 1. 构造 prompt 的工厂函数
#    根据模型路径名称决定使用 Chat 模板还是 Base 格式
# =====================================================
def get_prompt_builder(model_path, tokenizer):
    name = model_path.lower()

    # 针对指令微调 (Instruct) 模型或 Llama 系列
    # 使用 tokenizer 自带的 apply_chat_template 功能
    if "instruct" in name or "llama" in name or"deepseek" in name:
        def build_prompt(problem: str):
            messages = [
                {"role": "user", "content": f"Q: {problem}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"}
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        return build_prompt

    # 针对 DeepSeek 系列 (Base 模型处理)
    # if "deepseek" in name:
    #     def build_prompt(problem: str):
    #         return f"Q: {problem}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"
    #     return build_prompt

    # 针对 Qwen 系列 (Base 模型处理)
    if "qwen" in name:
        def build_prompt(problem: str):
            return f"Q: {problem}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"
        return build_prompt

    # 默认兜底格式
    def build_prompt(problem: str):
        return f"Q: {problem}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"
    
    return build_prompt


# =====================================================
# 2. 主流程
# =====================================================
def main(args):
    # ----------------------------
    # 1. 加载 Tokenizer
    # ----------------------------
    print(f"Loading tokenizer: {args.model_path}")
    # 注意：某些旧模型可能需要 use_fast=False，新模型通常 use_fast=True 更快
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, trust_remote_code=True)

    # ----------------------------
    # 2. 获取 Prompt 构造函数
    # ----------------------------
    build_prompt = get_prompt_builder(args.model_path, tokenizer)

    # ----------------------------
    # 3. 加载数据集
    # ----------------------------
    print(f"Loading dataset: {args.dataset_name} ({args.split})")
    dataset = load_dataset(args.dataset_name, split=args.split)

    # 如果指定了最大样本数，进行截断
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Total problems to process: {len(dataset)}")

    # ----------------------------
    # 4. 批量构造 Prompts
    # ----------------------------
    prompts = [build_prompt(item["problem"]) for item in dataset]

    # ----------------------------
    # 5. 设置采样参数 (SamplingParams)
    # ----------------------------
    sampling_params = SamplingParams(
        n=args.num_return_sequences,    # 每个问题生成的回答数量 (Pass@k)
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        # 设置停止词，防止模型无限生成或生成多轮对话
        stop=["<|eot_id|>", "</s>", "Q:", "<|im_end|>"],
    )

    # ----------------------------
    # 6. 初始化 vLLM 引擎
    # ----------------------------
    print(f"Initializing vLLM engine for model {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=torch.cuda.device_count() or 1, # 自动使用所有可用 GPU
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=0.90, # 可根据显存情况调整，默认0.9
    )

    # ----------------------------
    # 7. 执行推理
    # ----------------------------
    print("Starting generation...")
    request_outputs = llm.generate(prompts, sampling_params)

    # ----------------------------
    # 8. 保存结果到 JSONL
    # ----------------------------
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for i, request_output in enumerate(request_outputs):
            original_item = dataset[i]
            
            # 获取所有生成的候选答案（例如 n=4 时有 4 个答案）
            responses = [out.text for out in request_output.outputs]

            # 优先获取 answer 字段，如果没有则获取 solution 字段
            ground_truth = original_item.get("answer", original_item.get("solution", None))

            record = {
                "problem": original_item["problem"],
                "answer": ground_truth,
                "responses": responses,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ----------------------------
    # 9. 打印运行摘要
    # ----------------------------
    print("=== Generation Finished ===")
    print(f"Model path           : {args.model_path}")
    print(f"Max tokens           : {args.max_tokens}")
    print(f"Temperature          : {args.temperature}")
    print(f"Return sequences (n) : {args.num_return_sequences}")
    print(f"Output file          : {args.output_file}")
    print("===========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 vLLM 高效生成 N 条候选回答")
    
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/MATH-500", help="HuggingFace 数据集名称")
    parser.add_argument("--split", type=str, default="test", help="数据集划分 split")
    parser.add_argument("--model_path", type=str, required=True, help="本地模型路径或 HF 模型 ID")
    parser.add_argument("--output_file", type=str, default="math500_candidates.jsonl", help="输出文件路径")
    
    # 生成超参数
    parser.add_argument("--num_return_sequences", type=int, default=4, help="每个问题生成的回答数量")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    
    # 调试用参数
    parser.add_argument("--max_samples", type=int, default=None, help="仅处理前 N 个样本进行测试")
    
    args = parser.parse_args()
    main(args)