"""
Response-Certainty method implementation
"""

import re
import json
import time
import torch
import os
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils.common import (
    extract_model_answer, 
    is_correct_answer,
    generate_with_transformers
)


def calculate_response_confidence(prompt_ids, response_ids, logits, tokenizer):
    """
    使用完整概率分布计算整个响应的置信度（基于token的负熵）。
    
    Args:
        prompt_ids: 输入提示的token IDs
        response_ids: 生成响应的token IDs
        logits: 模型输出的logits
        tokenizer: 模型的分词器
        
    Returns:
        float: 整个响应的置信度分数
    """
    prompt_length = len(prompt_ids)
    # -1 因为 logits[t] 预测的是下一个 token
    response_logits = logits[prompt_length - 1:-1, :]
    
    # 计算每个位置的log概率
    log_probs = F.log_softmax(response_logits, dim=-1)
    
    # 计算token级别的置信度（负熵）
    token_confidences = []
    
    for i in range(len(response_ids)):
        if i < len(log_probs):
            token_log_probs = log_probs[i]
            # Token置信度：负的平均对数概率（熵的近似）
            token_confidence = -torch.mean(token_log_probs).item()
            token_confidences.append(token_confidence)
    
    # 计算平均token置信度作为整个响应的置信度
    avg_token_confidence = sum(token_confidences) / len(token_confidences) if token_confidences else 0.0
    
    return avg_token_confidence


def Response_Certainty_Selection(dataset, config, model, tokenizer, device,
                                 N=4, save_results=False):
    """
    评估流程：多样本生成 -> 整体置信度打分 -> 选最优 -> 保存为 question/answer/extracted_answer 三字段。
    """
    table = []
    n_true_ans = 0
    n_samples = 0
    start = time.time()
    index = 0

    progress_bar = tqdm(enumerate(dataset), desc="Processing")
    for i, data in progress_bar:
        model_answer = ""

        # 提取真值答案（GSM8K样式 #### <ans>）
        match = re.search(r'####\s*(.+)', data['answer'])
        if match:
            true_answer = match.group(1).strip()
        else:
            true_answer = data['answer'].strip()

        question = data['question']

        pro_prompt = """You are a math reasoning assistant. Solve each question carefully, showing clear step-by-step reasoning before giving the final numeric answer.  
            End each solution with the final numeric answer as: #### {final_number}
            
            Example 1:
            Q: A bakery sells muffins for $3 each and cupcakes for $2 each. If a customer buys 4 muffins and 6 cupcakes, how much does she spend in total?
            A: ## Step1: Each muffin costs $3, and she buys 4 muffins, so muffin cost = 4 × 3 = $12.  
            ## Step2: Each cupcake costs $2, and she buys 6 cupcakes, so cupcake cost = 6 × 2 = $12.  
            ## Step3: Total cost = 12 + 12 = $24.  
            #### 24
            
            Example 2:
            Q: A bus can carry 40 passengers. If there are 8 buses, and 10 seats are empty on each bus, how many passengers are on all the buses combined?
            A: ## Step1: Each bus has 40 seats, but 10 are empty, so there are 40 - 10 = 30 passengers per bus.  
            ## Step2: With 8 buses, total passengers = 8 × 30 = 240.  
            #### 240
            
            Example 3:
            Q: There are 5 boxes, and each box contains 12 apples. If 15 apples are eaten, how many apples remain?
            A: ## Step1: Total apples initially = 5 × 12 = 60.  
            ## Step2: After 15 apples are eaten, remaining = 60 - 15 = 45.  
            #### 45
            
            Example 4:
            Q: A train travels 80 miles in 2 hours. What is its average speed in miles per hour?
            A: ## Step1: Speed = distance ÷ time = 80 ÷ 2 = 40 mph.  
            #### 40
            
            Example 5:
            Q: A rectangle has a length of 10 cm and width of 4 cm. What is its area?
            A: ## Step1: Area = length × width = 10 × 4 = 40 square cm.  
            #### 40
            
            Example 6:
            Q: Sarah buys 3 notebooks costing $5 each and 2 pens costing $2 each. How much does she spend in total?
            A: ## Step1: Notebook cost = 3 × 5 = $15.  
            ## Step2: Pen cost = 2 × 2 = $4.  
            ## Step3: Total = 15 + 4 = $19.  
            #### 19
            
            Example 7:
            Q: A tank holds 500 liters of water. If 120 liters are drained out and 80 liters are added, how much water is now in the tank?
            A: ## Step1: Start with 500 liters.  
            ## Step2: Drain 120 liters → 500 - 120 = 380 liters.  
            ## Step3: Add 80 liters → 380 + 80 = 460 liters.  
            #### 460
            
            Example 8:
            Q: John reads 20 pages of a book each day. If the book has 180 pages, how many days will it take him to finish it?
            A: ## Step1: Days = total pages ÷ pages per day = 180 ÷ 20 = 9 days.  
            #### 9
            
            Now, solve the following problem step by step and end with the numeric answer in the same format:

            """
        
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{pro_prompt}Q: {question}\n\nA:"}],
            tokenize=False, add_generation_prompt=True
        )

        # 生成 N 个候选
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

        # 计算整个响应的置信度
        response_confidence_scores = []
        valid_candidates = []
        candidate_answers = []

        for candidate in candidates:
            try:
                response_confidence = calculate_response_confidence(
                    candidate["prompt_ids"],
                    candidate["tokens"],
                    candidate["logits"],
                    tokenizer
                )
                response_confidence_scores.append(response_confidence)
                valid_candidates.append(candidate)

                response_text_dbg = tokenizer.decode(candidate["tokens"], skip_special_tokens=False)
                extracted_answer = extract_model_answer(response_text_dbg)
                candidate_answers.append(extracted_answer)
            except Exception as e:
                print(f"Error calculating response confidence for candidate: {e}")
                response_confidence_scores.append(float('-inf'))
                valid_candidates.append(candidate)
                candidate_answers.append("")

        if not response_confidence_scores or all(score == float('-inf') for score in response_confidence_scores):
            print(f"No valid candidates for sample {i}")
            continue

        # 选择置信度最高的响应
        best_index = response_confidence_scores.index(max(response_confidence_scores))
        best_candidate = valid_candidates[best_index]

        # 解码最佳响应
        response_text = tokenizer.decode(best_candidate["tokens"], skip_special_tokens=False)

        # 清理控制标记（注意不要用含空分支的正则）
        cleaned_text = re.sub(r'<\|eot_id\|>', '', response_text)
        cleaned_text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', cleaned_text, flags=re.DOTALL)
        cleaned_text = re.sub(r'<\|.*?\|>', '', cleaned_text)
        cleaned_text = cleaned_text.replace("<|end_of_text|>", "")
        cleaned_text = cleaned_text.replace("<|end_of_sentence|>", "")
        cleaned_text = cleaned_text.replace("</s>", "")
        cleaned_text = re.sub(r'<｜end▁of▁sentence｜>', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # 直接从模型输出中提取答案
        model_answer = extract_model_answer(response_text)
 
        # 更新统计
        n_samples += 1
        if is_correct_answer(model_answer, true_answer):
            n_true_ans += 1

        # 保存为三字段格式
        table.append({
            "question": question,
            "answer": cleaned_text,
            "max_confidence": response_confidence_scores[best_index],
            "correct": is_correct_answer(model_answer, true_answer)
        })
        index += 1

        # 更新进度条显示
        acc_display = f"{(n_true_ans / n_samples):.4f}" if n_samples > 0 else "0.0000"
        progress_bar.set_postfix(accuracy=acc_display)

        # 前几个样本的调试输出
        if i < 10:
            print(f"\n--- 样本 {i+1} 调试信息 ---")
            print(f"问题: {question}")
            print(f"真实答案: {true_answer}")
            print(f"模型答案(提取): {model_answer}")
            print(f"清理后文本(保存的 answer): {cleaned_text}")
            print(f"最高置信度: {response_confidence_scores[best_index]:.4f}")
            print(f"是否正确: {is_correct_answer(model_answer, true_answer)}")
            print("--- 结束调试信息 ---\n")

    # 打印评估结果
    end = time.time()
    accuracy = n_true_ans / n_samples if n_samples > 0 else 0
    print("########################################################################################")
    print(f"Accuracy of Model@{N}: {accuracy:.4f}.")
    print(f"Total samples: {n_samples}, Correct: {n_true_ans}")
    print(f"Elapsed time: {end - start:.2f} secs.")
    print("########################################################################################")

    # 写入JSON（每条为 {question, answer, extracted_answer}）
    if save_results:
        os.makedirs("./TTT_data", exist_ok=True)
        output_file = f"./TTT_data/Self_Certainty_Trajectorylevel_best_of_{N}_Qwen7B_GSM8K.json"
        with open(output_file, mode="w", encoding="utf-8") as file:
            json.dump({
                "results": table,
                "accuracy": accuracy
            }, file, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_file}")