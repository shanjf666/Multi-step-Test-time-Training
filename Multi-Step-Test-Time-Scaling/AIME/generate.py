"""
generate.py
python generate.py --model_path meta-llama/Llama-3.2-1B-Instruct --num_return_sequences 1 --temperature 0.1 --max_tokens 1024 --output_file math500_candidates.jsonl
python generate.py --model_path meta-llama/Llama-3.2-1B-Instruct --num_return_sequences 4 --temperature 0.1 --max_tokens 1024 --output_file math500_candidates_0.jsonl
python generate.py --model_path /root/autodl-tmp/data/models/modelscope_cache/models/lijia321/llama_first --num_return_sequences 4 --temperature 0.1 --max_tokens 1024 --output_file math500_candidates_1.jsonl
python generate.py --model_path /root/autodl-tmp/data/models/modelscope_cache/models/lijia321/llama_second --num_return_sequences 4 --temperature 0.1 --max_tokens 1024 --output_file math500_candidates_2.jsonl
"""
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse

# =====================================================
# 1. 构造 prompt 的工厂函数（一次判断，多次使用）
# =====================================================
def get_prompt_builder(model_path, tokenizer):
    name = model_path.lower()

    # Chat-template 模型
    if "instruct" in name or "llama" in name:
        def build_prompt(problem: str):
            return tokenizer.apply_chat_template(
                [{"role": "user",
                  "content": f"Q: {problem}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"}],
                tokenize=False,
                add_generation_prompt=True
            )
        return build_prompt

    # DeepSeek
    if "deepseek" in name:
        def build_prompt(problem: str):
            return f"Q: {problem}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"
        return build_prompt

    # 普通 Qwen
    if "qwen" in name:
        def build_prompt(problem: str):
            return f"Q: {problem}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"
        return build_prompt

    # 默认
    def build_prompt(problem: str):
        return f"Q: {problem}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"
    return build_prompt


# =====================================================
# 2. 主流程
# =====================================================
def main(args):
    # ----------------------------
    # 加载 tokenizer
    # ----------------------------
    print(f"Loading tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)

    # ----------------------------
    # 创建 prompt 构造器
    # ----------------------------
    build_prompt = get_prompt_builder(args.model_path, tokenizer)

    # ----------------------------
    # 加载数据集
    # ----------------------------
    print(f"Loading dataset: {args.dataset_name} ({args.split})")
    # 注意：HuggingFaceH4/aime_2024 的字段通常也是 'problem' 和 'solution'/'answer'，
    # 所以下面的逻辑通常不需要改动，但要确保 split 正确。
    dataset = load_dataset(args.dataset_name, split=args.split)

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Total problems to process: {len(dataset)}")

    # ----------------------------
    # 构造 prompts（使用 build_prompt）
    # ----------------------------
    prompts = [build_prompt(item["problem"]) for item in dataset]

    # ----------------------------
    # 生成参数
    # ----------------------------
    sampling_params = SamplingParams(
        n=args.num_return_sequences,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        stop=["<|eot_id|>", "</s>", "Q:"],
    )

    # ----------------------------
    # 初始化 vLLM
    # ----------------------------
    print(f"Initializing vLLM engine for model {args.model_path}")
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=torch.cuda.device_count() or 1,
        trust_remote_code=True,
        dtype="auto",
    )

    # ----------------------------
    # 开始生成
    # ----------------------------
    print("Starting generation...")
    request_outputs = llm.generate(prompts, sampling_params)

    # ----------------------------
    # 保存输出
    # ----------------------------
    print(f"Saving results to {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for i, request_output in enumerate(request_outputs):
            original_item = dataset[i]

            responses = [out.text for out in request_output.outputs]

            # 兼容不同数据集的 key，AIME 2024 通常也有 answer 或 solution 字段
            record = {
                "problem": original_item["problem"],
                "answer": original_item.get("answer", original_item.get("solution", None)),
                "responses": responses,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ----------------------------
    # 输出生成参数信息
    # ----------------------------
    print("=== Generation Finished ===")
    print(f"model_path           : {args.model_path}")
    print(f"dataset              : {args.dataset_name}")
    print(f"max_tokens           : {args.max_tokens}")
    print(f"temperature          : {args.temperature}")
    print(f"num_return_sequences : {args.num_return_sequences}")
    print("===========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 vLLM 高效生成 N 条候选回答（适配 AIME 2024）")
    
    # --- 修改区域 Start ---
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/aime_2024", help="默认数据集改为 AIME 2024")
    # AIME 2024 在 HF 上通常只有 'train' split (因为它是一个很小的评测集)，如果是 'test' 可能会报错，这里改为 'train'
    parser.add_argument("--split", type=str, default="train", help="AIME 2024 默认 split 通常为 train")
    parser.add_argument("--output_file", type=str, default="aime2024_candidates.jsonl")
    # --- 修改区域 End ---

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_return_sequences", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048) # 数学题建议稍微调大一点 token 限制
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    
    args = parser.parse_args()
    main(args)