"""
export OPENAI_API_KEY='sk-szeemiuaxtqymzkkpaezgadntxpajxwyhrzofzsodywqngqz'
python base.py --model_path Qwen/Qwen2.5-Math-7B-Instruct
"""


import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# 禁用某些可能导致计算不一致的优化选项，确保结果的可重现性
os.environ["TRITON_ALLOW_MMA"] = "0"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from core.config import Config
from methods.response_certainty import Response_Certainty_Selection


def main():
    # 创建配置对象
    config = Config()

    # 命令行参数
    parser = argparse.ArgumentParser(description="基于整个回答置信度的评估方法")

    # 生成相关
    parser.add_argument("--method", default="response-certainty", 
                        choices=["response-certainty"], 
                        help="选择评估方法: response-certainty")
    parser.add_argument("--n_repetitive_sampling", default=4, type=int, help="为每个问题生成的解决方案数量")
    parser.add_argument("--temperature", default=0.7, type=float, help="生成时的温度参数，控制随机性")
    parser.add_argument("--top_p", default=1.0, type=float, help="Top-p采样参数")
    parser.add_argument("--model_path", default="meta-llama/Llama-3.2-1B-Instruct", help="基础模型路径")
    parser.add_argument("--save_to_json", default=True, action="store_true", help="是否将结果保存到JSON文件")
    parser.add_argument("--dataset_repo_name", default="HuggingFaceH4/MATH-500", help="数据集仓库名称")
    parser.add_argument("--max_tokens", default=512, type=int, help="最大生成标记数")
    parser.add_argument("--subset_size", default=None, type=int, help="测试子集大小（用于快速测试）")

    args = parser.parse_args()

    # 将解析的参数赋值给配置对象
    config.n = int(args.n_repetitive_sampling)
    config.temperature = float(args.temperature)
    config.top_p = float(args.top_p)
    config.max_tokens = int(args.max_tokens)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载基础语言模型
    print("正在加载语言模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 添加 padding token 如果不存在
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    print("语言模型加载完成!")

    # 加载数据集
    print("正在加载数据集...")
    dataset = load_dataset(args.dataset_repo_name, 'default', split="test")

    # 测试子集
    if args.subset_size:
        dataset = dataset.select(range(min(args.subset_size, len(dataset))))

    print(f"数据集加载完成，共{len(dataset)}个样本")

    # 执行评估 + 保存为三字段
    print("开始执行基于整个回答置信度的评估...")
    
    if args.method == "response-certainty":
        Response_Certainty_Selection(
            dataset,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device=device,
            N=config.n,
            save_results=args.save_to_json
        )

    # 打印所有参数及其值
    print("########################################################################################")
    print("实验配置参数:")
    print("########################################################################################")
    for arg, value in vars(args).items():
        print(f"{arg:>25} ===> {value}")
    print("########################################################################################")


if __name__ == "__main__":
    main()