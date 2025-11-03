"""
Self-Eval method implementation
"""

import re
import json
import time
import torch
import os
from tqdm import tqdm
from utils.common import (
    parse_structured_steps, 
    extract_model_answer, 
    is_correct_answer,
    generate_with_transformers,
    calculate_step_confidence_with_self_eval,
    clean_latex_format
)
from utils.key_step_extractor import summarize_key_steps_openai


def Self_Eval_Selection(dataset, config, model, tokenizer, device,
                        N=4, save_results=False, lambda_weight=0.5,
                        key_api_client=None,
                        key_api_model="qwen3-max-preview",
                        key_api_temp=0.2):
    """
    基于步骤的置信度计算并选择最佳答案，使用API模型提取关键步骤后再提取答案
    """
    table = []
    n_true_ans = 0
    n_samples = 0
    start = time.time()
    index = 0
    
    progress_bar = tqdm(enumerate(dataset), desc="Processing")
    for i, data in progress_bar:
        model_answer = ""

        # 提取真值答案（MATH500样式）
        true_answer = clean_latex_format(data['answer'])

        question = data['problem']

        # 构建模型提示
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{config.system_prompt}Q: {question}\n\nA:"}],
            tokenize=False, add_generation_prompt=True
        )

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

        step_confidence_scores = []
        valid_candidates = []
        candidate_answers = []
        
        for candidate in candidates:
            try:
                step_confidence = calculate_step_confidence_with_self_eval(
                    candidate["prompt_ids"], 
                    candidate["tokens"], 
                    candidate["logits"], 
                    tokenizer,
                    lambda_weight=lambda_weight,
                    eval_model=model,
                    question=question,
                    device=device,
                    steps=parse_structured_steps(tokenizer.decode(candidate["tokens"], skip_special_tokens=True))
                )
                step_confidence_scores.append(step_confidence)
                valid_candidates.append(candidate)
                
                response_text = tokenizer.decode(candidate["tokens"], skip_special_tokens=False)
                extracted_answer = extract_model_answer(response_text)
                candidate_answers.append(extracted_answer)
                
            except Exception as e:
                print(f"Error calculating step confidence for candidate: {e}")
                step_confidence_scores.append(float('-inf'))
                valid_candidates.append(candidate)
                candidate_answers.append("")

        if not step_confidence_scores or all(score == float('-inf') for score in step_confidence_scores):
            print(f"No valid candidates for sample {i}")
            continue
            
        best_index = step_confidence_scores.index(max(step_confidence_scores))
        best_candidate = valid_candidates[best_index]
        
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
        cleaned_text = re.sub(r'<|endoftext|>', '', cleaned_text)
        
        # 使用 OpenAI 兼容接口抽取关键步骤
        key_step_text = ""
        try:
            key_step_text = summarize_key_steps_openai(
                client=key_api_client,
                model=key_api_model,
                reasoning_text=cleaned_text,
                temperature=key_api_temp
            )
        except Exception as e:
            print(f"[WARN] key-step extraction failed for sample {i}: {e}")
            key_step_text = ""

        if key_step_text.strip():
            model_answer = extract_model_answer(key_step_text)
        else:
            # 回退策略
            model_answer = extract_model_answer(response_text)
        
        # 更新统计
        n_samples += 1
        clean_key_step_text = clean_latex_format(key_step_text)
        if true_answer in clean_key_step_text[-30:] or is_correct_answer(model_answer, true_answer):
            n_true_ans += 1
        
        # 保存结果
        table.append({
            "question": question,
            "answer": cleaned_text,
            "gpt_response": key_step_text
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
            print(f"关键步骤(gpt_response): {key_step_text}")
            print(f"模型答案(提取): {model_answer}")
            print(f"是否正确: {true_answer in clean_key_step_text[-30:] or is_correct_answer(model_answer, true_answer)}")
            print("--- 结束调试信息 ---\n")
    
    # 计算并打印评估结果
    end = time.time()
    accuracy = n_true_ans/n_samples if n_samples > 0 else 0
    print("########################################################################################")
    print(f"Accuracy of Model@{N}: {accuracy:.4f}.")
    print(f"Total samples: {n_samples}, Correct: {n_true_ans}")
    print(f"Elapsed time: {end-start:.2f} secs.")
    print("########################################################################################")

    # 如果需要保存结果，则写入JSON文件
    if save_results:
        os.makedirs("./TTT_data", exist_ok=True)
        output_file = f"./TTT_data/Self_Eval_best_of_{N}_lambda_{lambda_weight}_Qwen7B_MATH500.json"
        with open(output_file, mode="w", encoding="utf-8") as file:
            json.dump({
                "results": table,
                "accuracy": accuracy
            }, file, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_file}")