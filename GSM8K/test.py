import json
import os
from datasets import load_dataset
from openai import OpenAI
from utils.key_step_extractor import summarize_key_steps_openai
from tqdm import tqdm


def main():
    # 初始化关键步骤提取的 OpenAI 兼容客户端
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未设置。请先 export OPENAI_API_KEY='your_api_key'")
    key_client = OpenAI(api_key=api_key, base_url="https://api.siliconflow.cn/v1")

    # 加载数据集
    print("正在加载数据集...")
    dataset = load_dataset("openai/gsm8k", 'main', split="test")
    print(dataset[0])

    # 打开一个 jsonl 文件进行写入
    output_file = "./TTT_data/GSM8K.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        progress_bar = tqdm(enumerate(dataset), desc="Processing")
        for i, data in progress_bar:
            question = data['question']  # 获取问题文本
            cleaned_text = data['answer']  # 获取答案文本
            
            # 仅使用 OpenAI 兼容接口抽取关键步骤
            key_step_text = ""
            try:
                key_step_text = summarize_key_steps_openai(
                    client=key_client,
                    model="Qwen/Qwen2.5-7B-Instruct",
                    reasoning_text=cleaned_text,
                    temperature=0.2
                )
            except Exception as e:
                print(f"[WARN] key-step extraction failed for sample {i}:{e}")
                key_step_text = ""
            
            # 将结果以字典形式写入文件
            output_data = {
                "question": question,
                "answer": cleaned_text,
                "gpt_response": key_step_text
            }
            
            # 将字典转为 JSON 字符串并写入 jsonl 文件
            f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

    print(f"处理完成，数据已保存到 {output_file} 文件中。")


if __name__ == "__main__":
    main()
    