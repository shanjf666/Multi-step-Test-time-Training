"""
python infer.py --model_path meta-llama/Llama-3.2-1B-Instruct
CUDA_VISIBLE_DEVICES=1 python infer.py --model_path Qwen/Qwen2.5-Math-1.5B
python infer.py --model_path Qwen/Qwen2.5-Math-1.5B-Instruct
python infer.py --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
"""
import argparse
import subprocess
import os
import json
import datetime

def run_pipeline(model_path, num_return_sequences=4, temperature=0.1, max_tokens=1024, alpha=0.4):
    # === 创建输出目录 ===
    os.makedirs("results", exist_ok=True)

    # === 时间戳 & 模型名 ===
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = os.path.basename(model_path.rstrip("/"))

    # === 中间与最终文件路径 ===
    candidates_file = f"results/aime2024_candidates_{model_name}_{timestamp}.jsonl"
    scored_file = f"results/aime2024_best_{model_name}_{timestamp}.jsonl"

    # === Step 1. 调用 generate.py 生成候选答案 ===
    print(f"[1/2] Generating candidates -> {candidates_file}")
    subprocess.run([
        "python", "generate.py",
        "--model_path", model_path,
        "--num_return_sequences", str(num_return_sequences),
        "--temperature", str(temperature),
        "--max_tokens", str(max_tokens),
        "--output_file", candidates_file
    ], check=True)

    # === Step 2. 调用 evaluate.py 打分并选出最优回答 ===
    print(f"[2/2] Evaluating and selecting best answers -> {scored_file}")
    subprocess.run([
        "python", "evaluate.py",
        "--input_file", candidates_file,
        "--output_file", scored_file,
        "--model_dir", model_path,
        "--alpha", str(alpha)
    ], check=True)

    # === Step 3. 输出总结 ===
    print(f"✅ Done! Final results saved to: {scored_file}")
    print("=" * 60)

    # 可选：打印前几个结果做验证
    try:
        with open(scored_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f.readlines()[:3]]
        print(json.dumps(lines, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Warning: Could not preview output file ({e})")

    return scored_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一调用 generate.py + evaluate.py 的一键推理管线")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径（本地或HF模型名）")
    parser.add_argument("--num_return_sequences", type=int, default=4, help="每题生成候选数量")
    parser.add_argument("--temperature", type=float, default=0.5, help="采样温度")
    parser.add_argument("--max_tokens", type=int, default=3072, help="生成最大长度")
    parser.add_argument("--alpha", type=float, default=0.0, help="evaluate.py 的 self-certainty 加权系数")
    args = parser.parse_args()

    run_pipeline(
        model_path=args.model_path,
        num_return_sequences=args.num_return_sequences,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        alpha=args.alpha
    )
