"""
Baseline method implementation
"""


import re
import json
import time
import torch
import os
from tqdm import tqdm
from utils.common import (
    extract_model_answer, 
    is_correct_answer,
    generate_with_transformers,
    clean_latex_format
)


def baseline_evaluation(dataset, config, model, tokenizer, device, save_results=False):
    """
    Baseline评估函数 - 每个问题只生成1次答案，不使用self-certainty评分
    
    Args:
        dataset: 测试数据集
        config: 配置对象
        model: Transformers模型
        tokenizer: 分词器
        device: 计算设备
        save_results: 是否保存结果到JSON文件
    """
    # 初始化结果存储列表和统计变量
    table = []  # 存储详细结果
    n_true_ans = 0  # 正确答案计数
    n_samples = 0  # 样本总数
    start = time.time()  # 记录开始时间
    index = 0  # 用于记录保存到文件的结果索引
    
    # 创建进度条，遍历数据集
    progress_bar = tqdm(enumerate(dataset), desc="Processing")
    for i, data in progress_bar:
        model_answer = ""  # 模型答案
        # 提取真值答案（MATH500样式）
        true_answer = clean_latex_format(data['answer'])

        question = data['problem']

        # 构建模型提示
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{config.system_prompt}Q: {question}\n\nA:"}],
            tokenize=False, add_generation_prompt=True
        )

        # 生成1个候选答案
        try:
            candidate = generate_with_transformers(
                model, tokenizer, prompt, device, 
                temperature=config.temperature, 
                max_tokens=config.max_tokens,
                num_return_sequences=1
            )[0]  # 只取第一个结果
        except Exception as e:
            print(f"Error generating answer for sample {i}: {e}")
            continue

        # 解码响应
        response_text = tokenizer.decode(candidate["tokens"], skip_special_tokens=False)
        
        # 提取模型答案
        model_answer = extract_model_answer(response_text)
        
        # 检查答案是否正确
        n_samples += 1
        clean_response_text = clean_latex_format(response_text)
        if true_answer in clean_response_text[-10:] or is_correct_answer(model_answer, true_answer):
            n_true_ans += 1

        # 清理控制标记（注意不要用含空分支的正则）
        cleaned_text = re.sub(r'<\|eot_id\|>', '', response_text)
        cleaned_text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', cleaned_text, flags=re.DOTALL)
        cleaned_text = re.sub(r'<\|.*?\|>', '', cleaned_text)
        cleaned_text = cleaned_text.replace("<|end_of_text|>", "")
        cleaned_text = cleaned_text.replace("<|end_of_sentence|>", "")
        cleaned_text = cleaned_text.replace("</s>", "")
        cleaned_text = re.sub(r'<｜end▁of▁sentence｜>', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # 保存结果
        table.append({
            "question": question,
            "answer": cleaned_text,
        })
        index += 1
                
        # 更新进度条显示
        acc_display = f"{(n_true_ans / n_samples):.4f}" if n_samples > 0 else "0.0000"
        progress_bar.set_postfix(accuracy=acc_display)
        
        # 调试信息（前几个样本）
        if i < 10:
            print(f"\n--- 样本 {i+1} 调试信息 ---")
            print(f"问题: {question}")
            print(f"答案：{data['answer']}")
            print(f"提取后的答案: {true_answer}")
            print(f"清理后文本(保存的 answer): {clean_response_text}")
            print(f"模型答案(提取): {model_answer}")
            print(f"是否正确: {true_answer in clean_response_text[-10:] or is_correct_answer(model_answer, true_answer)}")
            print("--- 结束调试信息 ---\n")
    
    # 计算并打印评估结果
    end = time.time()
    accuracy = n_true_ans/n_samples if n_samples > 0 else 0
    print("########################################################################################")
    print(f"Baseline Accuracy: {accuracy:.4f}.")
    print(f"Total samples: {n_samples}, Correct: {n_true_ans}")
    print(f"Elapsed time: {end-start:.2f} secs.")
    print("########################################################################################")

    # 如果需要保存结果，则写入JSON文件
    if save_results:
        os.makedirs("./TTT_data", exist_ok=True)
        output_file = f"./TTT_data/baseline_deepseek_7b.json"
        with open(output_file, mode="w", encoding="utf-8") as file:
            json.dump({
                "results": table,
                "accuracy": accuracy
            }, file, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_file}")