"""
export OPENAI_API_KEY='sk-szeemiuaxtqymzkkpaezgadntxpajxwyhrzofzsodywqngqz'
python main.py --method self-certainty --model_path meta-llama/Llama-3.2-1B-Instruct --lambda_weight 0.5 --n_repetitive_sampling 4 --max_tokens 2048
python main.py --method self-consistency --model_path /root/autodl-tmp/multi-TTT_test --lambda_weight 0.5 --subset_size 100 --max_tokens 1024
"""

import os
import torch
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

# 禁用某些可能导致计算不一致的优化选项，确保结果的可重现性
os.environ["TRITON_ALLOW_MMA"] = "0"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 将相对导入改为绝对导入
from core.config import Config
from methods.self_certainty import Self_Certainty_Selection
from methods.self_eval import Self_Eval_Selection
from methods.coe_c import CoE_C_Selection
from methods.baseline import baseline_evaluation
from methods.self_consistency import Self_Consistency_Selection


def main():
    # 创建配置对象
    config = Config()

    # 命令行参数
    parser = argparse.ArgumentParser(description="基于步骤置信度的评估方法（Transformers + GPT关键步骤提取）")


def main():
    # 创建配置对象
    config = Config()

    # 命令行参数
    parser = argparse.ArgumentParser(description="基于步骤置信度的评估方法（Transformers + GPT关键步骤提取）")

    # 生成相关
    parser.add_argument("--method", default="self-certainty", 
                        choices=["self-certainty", "self-eval", "coe-c", "baseline", "self-consistency"], 
                        help="选择评估方法: self-certainty, self-eval, coe-c, baseline, self-consistency")
    parser.add_argument("--n_repetitive_sampling", default=4, type=int, help="为每个问题生成的解决方案数量")
    parser.add_argument("--temperature", default=0.1, type=float, help="生成时的温度参数，控制随机性")
    parser.add_argument("--top_p", default=1.0, type=float, help="Top-p采样参数")
    parser.add_argument("--model_path", default="meta-llama/Llama-3.2-1B-Instruct", help="基础模型路径")
    parser.add_argument("--save_to_json", default=True, action="store_true", help="是否将结果保存到JSON文件")
    parser.add_argument("--dataset_repo_name", default="gneubig/aime-1983-2024", help="数据集仓库名称")
    parser.add_argument("--max_tokens", default=512, type=int, help="最大生成标记数")
    parser.add_argument("--subset_size", default=None, type=int, help="测试子集大小（用于快速测试）")
    parser.add_argument("--lambda_weight", default=0.5, type=float,
                        help="置信度计算中的lambda权重参数，用于平衡token置信度和步骤概率")

    # 关键步骤提取（OpenAI 兼容接口，仅用 API）
    parser.add_argument("--key_step_api_base", default="https://api.siliconflow.cn/v1",
                        help="OpenAI-compatible base URL for key-step extraction")
    parser.add_argument("--key_step_api_model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name for key-step extraction")
    parser.add_argument("--key_step_temperature", default=0.2, type=float,
                        help="Temperature for key-step extraction")

    args = parser.parse_args()

    # 将解析的参数赋值给配置对象
    config.n = int(args.n_repetitive_sampling)
    config.temperature = float(args.temperature)
    config.top_p = float(args.top_p)
    config.max_tokens = int(args.max_tokens)
    config.lambda_weight = float(args.lambda_weight)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化关键步骤提取的 OpenAI 兼容客户端
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未设置。请先 export OPENAI_API_KEY='your_api_key'")
    key_client = OpenAI(api_key=api_key, base_url=args.key_step_api_base)

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
    dataset = load_dataset(args.dataset_repo_name, 'default', split="train")
    print(dataset[0])

    # 测试子集
    if args.subset_size:
        dataset = dataset.select(range(min(args.subset_size, len(dataset))))

    print(f"数据集加载完成，共{len(dataset)}个样本")

    # 执行评估 + 保存为三字段
    print("开始执行基于步骤的置信度评估...")
    
    if args.method == "self-certainty":
        Self_Certainty_Selection(
            dataset,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device=device,
            N=config.n,
            save_results=args.save_to_json,
            lambda_weight=config.lambda_weight,
            key_api_client=key_client,
            key_api_model=args.key_step_api_model,
            key_api_temp=args.key_step_temperature
        )
    elif args.method == "self-eval":
        Self_Eval_Selection(
            dataset,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device=device,
            N=config.n,
            save_results=args.save_to_json,
            lambda_weight=config.lambda_weight,
            key_api_client=key_client,
            key_api_model=args.key_step_api_model,
            key_api_temp=args.key_step_temperature
        )
    elif args.method == "coe-c":
        CoE_C_Selection(
            dataset,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device=device,
            N=config.n,
            save_results=args.save_to_json,
            lambda_weight=config.lambda_weight,
            key_api_client=key_client,
            key_api_model=args.key_step_api_model,
            key_api_temp=args.key_step_temperature
        )
    elif args.method == "baseline":
        baseline_evaluation(
            dataset,
            config=config,
            model=model,
            tokenizer=tokenizer,
            device=device,
            save_results=args.save_to_json
        )
    elif args.method == "self-consistency":
        Self_Consistency_Selection(
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
