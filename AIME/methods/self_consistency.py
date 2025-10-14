"""
Self-Consistency method implementation
"""

import re
import json
import time
import torch
import os
from tqdm import tqdm
from collections import Counter
from utils.common import (
    extract_model_answer, 
    is_correct_answer,
    generate_with_transformers
)


def Self_Consistency_Selection(dataset, config, model, tokenizer, device, N=4, save_results=False):
    """
    基于Self-Consistency的方法选择最终答案
    
    Args:
        dataset: 测试数据集
        config: 配置对象
        model: Transformers模型
        tokenizer: 分词器
        device: 计算设备
        N: 为每个问题生成的解决方案数量
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
        # 使用正则表达式提取真实答案
        match = re.search(r'####\s*(.+)', data['answer'])
        if match:
            true_answer = match.group(1).strip()
        else:
            true_answer = data['answer'].strip()
            
        question = data['question']
        
        # 构建提示格式
        prompt = f"<|begin_of_text|>{config.system_prompt}\nQuestion: {question}\nAnswer:"

        # 生成N个候选答案
        try:
            candidates = generate_with_transformers(
                model, tokenizer, prompt, device, 
                temperature=config.temperature, 
                max_tokens=config.max_tokens, 
                num_return_sequences=N
            )
        except Exception as e:
            print(f"Error generating candidates for sample {i}: {e}")
            continue

        # 提取所有候选答案
        candidate_answers = []
        valid_candidates = []
        
        for candidate in candidates:
            try:
                extracted_answer = extract_model_answer(tokenizer.decode(candidate["tokens"], skip_special_tokens=False))
                candidate_answers.append(extracted_answer)
                valid_candidates.append(tokenizer.decode(candidate["tokens"], skip_special_tokens=False))
            except Exception as e:
                print(f"Error extracting answer from candidate: {e}")
                candidate_answers.append("")
                valid_candidates.append("")

        # 统计答案频率，选择出现最多的答案
        answer_counter = Counter(candidate_answers)
        # 移除空答案
        if "" in answer_counter:
            del answer_counter[""]
        
        if answer_counter:
            # 选择出现频率最高的答案
            model_answer = answer_counter.most_common(1)[0][0]
        else:
            model_answer = ""

        # 检查答案是否正确
        n_samples += 1
        if is_correct_answer(model_answer, true_answer):
            n_true_ans += 1

        # 保存结果
        table.append({
            "ID": index+1, 
            "model_input": question, 
            "outputs": valid_candidates,  # 保存所有生成的结果
            "extracted_answers": candidate_answers,  # 保存所有提取的答案
            "model_answer": model_answer,  
            "true_answer": true_answer,
            "answer_counts": dict(answer_counter),  # 保存答案统计
            "is_correct": is_correct_answer(model_answer, true_answer)
        })
        index += 1
                
        # 更新进度条显示
        progress_bar.set_postfix(accuracy=f"{n_true_ans/n_samples:.4f}")
        
        # 调试信息（前几个样本）
        if i < 10:
            print(f"\n--- 样本 {i+1} 调试信息 ---")
            print(f"问题: {question}")
            print(f"真实答案: {true_answer}")
            print(f"模型答案: {model_answer}")
            print(f"答案统计: {dict(answer_counter)}")
            print(f"是否正确: {is_correct_answer(model_answer, true_answer)}")
            print("--- 结束调试信息 ---\n")
    
    # 计算并打印评估结果
    end = time.time()
    accuracy = n_true_ans/n_samples if n_samples > 0 else 0
    print("########################################################################################")
    print(f"Accuracy of Model@{N} (Self-Consistency): {accuracy:.4f}.")
    print(f"Total samples: {n_samples}, Correct: {n_true_ans}")
    print(f"Elapsed time: {end-start:.2f} secs.")
    print("########################################################################################")

    # 如果需要保存结果，则写入JSON文件
    if save_results:
        os.makedirs("./TTT_data", exist_ok=True)
        output_file = f"./TTT_data/Best_of_{N}_Transformers_Self_Consistency_deepseek.json"
        with open(output_file, mode="w", encoding="utf-8") as file:
            json.dump(table, file, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_file}")