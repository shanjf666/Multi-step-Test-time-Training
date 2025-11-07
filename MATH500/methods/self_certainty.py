"""
Self-Certainty method implementation
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
    calculate_step_confidence_with_self_certainty,
    clean_latex_format,
    generate_with_transformers_batch
)
from utils.key_step_extractor import summarize_key_steps_openai

# def Self_Certainty_Selection(dataset, config, model, tokenizer, device,
#                              N=4, save_results=False, lambda_weight=0.5,
#                              batch_size=2, # 新增：批处理大小
#                              key_api_client=None,
#                              key_api_model="qwen3-max-preview",
#                              key_api_temp=0.2):
#     """
#     评估流程（批处理版）：批量生成 -> 逐个步骤置信度打分 -> 选最优 -> 保存。
#     完全保留了原有的评分、筛选和清洗逻辑。
#     """
#     table = []
#     n_true_ans = 0
#     n_samples = 0
#     start = time.time()
    
#     # 计算总批次数
#     num_batches = (len(dataset) + batch_size - 1) // batch_size
    
#     print(f"Starting evaluation with Batch Size = {batch_size}, Total Batches = {num_batches}")
#     progress_bar = tqdm(range(num_batches), desc="Processing Batches")

#     for batch_idx in progress_bar:
#         # --- 1. 准备当前批次的数据 ---
#         start_idx = batch_idx * batch_size
#         end_idx = min((batch_idx + 1) * batch_size, len(dataset))
        
#         # 【关键修复】：使用列表推导式逐行获取，避免 dataset切片返回列式存储导致的 TypeError
#         current_batch_data = [dataset[i] for i in range(start_idx, end_idx)]
        
#         batch_prompts = []
#         batch_meta = [] # 用于存储元数据，以便在生成后恢复对应关系

#         for local_i, data in enumerate(current_batch_data):
#             global_idx = start_idx + local_i
#             question = data['problem']
#             # 提前提取真值
#             true_answer = clean_latex_format(data['answer'])

#             # 构建模型提示 (保持和你原代码一致)
#             prompt = tokenizer.apply_chat_template(
#                 [{"role": "user", "content": f"Q: {question}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"}],
#                 tokenize=False, add_generation_prompt=True
#             )
            
#             batch_prompts.append(prompt)
#             # 保存元数据供后续使用
#             batch_meta.append({
#                 "global_idx": global_idx,
#                 "question": question,
#                 "true_answer": true_answer,
#                 "original_data": data
#             })

#         # --- 2. 批量生成 (调用 generate_with_transformers_batch) ---
#         try:
#             # 假设 generate_with_transformers_batch 返回 [Batch, N, Dict] 结构
#             batch_candidates_grouped = generate_with_transformers_batch(
#                 model, tokenizer, batch_prompts, device,
#                 temperature=config.temperature,
#                 max_tokens=config.max_tokens,
#                 num_return_sequences=N
#             )
#         except Exception as e:
#             print(f"Error generating for batch {batch_idx}: {e}")
#             # 如果整个批次生成失败，跳过以防崩溃
#             continue

#         # --- 3. 逐个处理批次内的结果 (逻辑与原版完全一致) ---
#         for local_i, candidates in enumerate(batch_candidates_grouped):
#             # 恢复当前样本的元数据
#             meta = batch_meta[local_i]
#             i = meta["global_idx"]
#             question = meta["question"]
#             true_answer = meta["true_answer"]
#             data_original = meta["original_data"]

#             step_confidence_scores = []
#             valid_candidates = []
#             # candidate_answers = [] 

#             for candidate in candidates:
#                 try:
#                     # 计算步骤置信度
#                     step_confidence = calculate_step_confidence_with_self_certainty(
#                         candidate["prompt_ids"],
#                         candidate["tokens"],
#                         candidate["logits"],
#                         tokenizer,
#                         lambda_weight=lambda_weight
#                     )
#                     step_confidence_scores.append(step_confidence)
#                     valid_candidates.append(candidate)

#                     # (可选) 调试用的中间提取，如果不需打印可以注释掉以提升速度
#                     # response_text_dbg = tokenizer.decode(candidate["tokens"], skip_special_tokens=False)
#                     # extracted_answer = extract_model_answer(response_text_dbg)
#                     # candidate_answers.append(extracted_answer)

#                 except Exception as e:
#                     # print(f"Error calculating confidence for sample {i}: {e}")
#                     step_confidence_scores.append(float('-inf'))
#                     valid_candidates.append(candidate)

#             # 如果所有候选都计算失败，跳过该样本
#             if not step_confidence_scores or all(score == float('-inf') for score in step_confidence_scores):
#                 print(f"No valid candidates for sample {i}")
#                 continue

#             # 选择置信度最高的响应
#             best_index = step_confidence_scores.index(max(step_confidence_scores))
#             best_candidate = valid_candidates[best_index]

#             # 解码最佳响应
#             response_text = tokenizer.decode(best_candidate["tokens"], skip_special_tokens=False)

#             # --- 清理逻辑 (保持原样) ---
#             cleaned_text = re.sub(r'<\|eot_id\|>', '', response_text)
#             cleaned_text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', cleaned_text, flags=re.DOTALL)
#             cleaned_text = re.sub(r'<\|.*?\|>', '', cleaned_text)
#             cleaned_text = cleaned_text.replace("<|end_of_text|>", "")
#             cleaned_text = cleaned_text.replace("<|end_of_sentence|>", "")
#             cleaned_text = cleaned_text.replace("</s>", "")
#             cleaned_text = re.sub(r'<｜end of sentence｜>', '', cleaned_text)
#             cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
#             cleaned_text = re.sub(r'<|endoftext|>', '', cleaned_text)

#             # 提取模型答案
#             model_answer = extract_model_answer(response_text)

#             # 更新统计
#             n_samples += 1
#             clean_key_step_text = clean_latex_format(cleaned_text)
#             model_answer_clean = clean_latex_format(model_answer)
            
#             is_correct = False
#             # 判断正确性 (保持原逻辑: 检查末尾或精确匹配)
#             if true_answer in clean_key_step_text[-30:] or is_correct_answer(model_answer_clean, true_answer):
#                 n_true_ans += 1
#                 is_correct = True

#             # 添加到结果表
#             table.append({
#                 "question": question,
#                 "answer": cleaned_text,
#                 # "gpt_response": key_step_text # 原代码已注释
#             })

#             # 更新进度条上的准确率显示
#             acc_display = f"{(n_true_ans / n_samples):.4f}" if n_samples > 0 else "0.0000"
#             progress_bar.set_postfix(accuracy=acc_display)

#             # --- 调试信息打印 (保持原样，仅打印前10个) ---
#             if i < 10:
#                 print(f"\n--- 样本 {i+1} 调试信息 ---")
#                 print(f"问题: {question}")
#                 print(f"答案：{data_original['answer']}")
#                 print(f"提取后的答案: {true_answer}")
#                 print(f"清理后文本(保存的 answer): {cleaned_text}")
#                 print(f"模型答案(提取): {model_answer_clean}")
#                 print(f"是否正确: {is_correct}")
#                 print("--- 结束调试信息 ---\n")


def Self_Certainty_Selection(dataset, config, model, tokenizer, device,
                             N=4, save_results=False, lambda_weight=0.5,
                             key_api_client=None,
                             key_api_model="qwen3-max-preview",
                             key_api_temp=0.2):
    """
    评估流程：多样本生成 -> 步骤置信度打分 -> 选最优 -> 保存为 question/answer/gpt_response 三字段。
    关键步骤仅通过 OpenAI 兼容接口提取（不做本地启发式）。
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
            [{"role": "user", "content": f"Q: {question}\nLet's think step by step and output the final answer within \\boxed{{}}\nA:"}],
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

        # 计算步骤置信度
        step_confidence_scores = []
        valid_candidates = []
        candidate_answers = []

        for candidate in candidates:
            try:
                step_confidence = calculate_step_confidence_with_self_certainty(
                    candidate["prompt_ids"],
                    candidate["tokens"],
                    candidate["logits"],
                    tokenizer,
                    lambda_weight=lambda_weight
                )
                step_confidence_scores.append(step_confidence)
                valid_candidates.append(candidate)

                response_text_dbg = tokenizer.decode(candidate["tokens"], skip_special_tokens=False)
                extracted_answer = extract_model_answer(response_text_dbg)
                candidate_answers.append(extracted_answer)
            except Exception as e:
                print(f"Error calculating step confidence for candidate: {e}")
                step_confidence_scores.append(float('-inf'))
                valid_candidates.append(candidate)
                candidate_answers.append("")

        if not step_confidence_scores or all(score == float('-inf') for score in step_confidence_scores):
            print(f"No valid candidates for sample {i}")
            continue

        # 选择置信度最高的响应
        best_index = step_confidence_scores.index(max(step_confidence_scores))
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
        cleaned_text = re.sub(r'<|endoftext|>', '', cleaned_text)

        # 仅使用 OpenAI 兼容接口抽取关键步骤
        # key_step_text = ""
        # try:
        #     key_step_text = summarize_key_steps_openai(
        #         client=key_api_client,
        #         model=key_api_model,
        #         reasoning_text=cleaned_text,
        #         temperature=key_api_temp
        #     )
        # except Exception as e:
        #     print(f"[WARN] key-step extraction failed for sample {i}: {e}")
        #     key_step_text = ""

        # if key_step_text.strip():
        #     model_answer = extract_model_answer(key_step_text)
        # else:
        #     # 回退策略（可按需只留一个回退）
        #     model_answer = extract_model_answer(response_text)
        
        model_answer = extract_model_answer(response_text)

        # 更新统计
        n_samples += 1
        # clean_key_step_text = clean_latex_format(key_step_text)
        clean_key_step_text = clean_latex_format(cleaned_text)
        model_answer = clean_latex_format(model_answer)
        if true_answer in clean_key_step_text[-30:] or is_correct_answer(model_answer, true_answer):
            n_true_ans += 1

        # 保存为三字段格式
        table.append({
            "question": question,
            "answer": cleaned_text,
            # "gpt_response": key_step_text
        })
        index += 1

        # 更新进度条显示
        acc_display = f"{(n_true_ans / n_samples):.4f}" if n_samples > 0 else "0.0000"
        progress_bar.set_postfix(accuracy=acc_display)

        # 前几个样本的调试输出
        if i < 10:
            print(f"\n--- 样本 {i+1} 调试信息 ---")
            print(f"问题: {question}")
            print(f"答案：{data['answer']}")
            print(f"提取后的答案: {true_answer}")
            print(f"清理后文本(保存的 answer): {cleaned_text}")
            # print(f"关键步骤(gpt_response): {key_step_text}")
            print(f"模型答案(提取): {model_answer}")
            # print(f"最高置信度: {step_confidence_scores[best_index]:.4f}")
            print(f"是否正确: {true_answer in clean_key_step_text[-30:] or is_correct_answer(model_answer, true_answer)}")
            
            print("--- 结束调试信息 ---\n")
                
    # 打印评估结果
    end = time.time()
    accuracy = n_true_ans / n_samples if n_samples > 0 else 0
    print("########################################################################################")
    print(f"Accuracy of Model@{N}: {accuracy:.4f}.")
    print(f"Total samples: {n_samples}, Correct: {n_true_ans}")
    print(f"Elapsed time: {end - start:.2f} secs.")
    print("########################################################################################")

    # 写入JSON（每条为 {question, answer, gpt_response}）
    if save_results:
        os.makedirs("./TTT_data", exist_ok=True)
        output_file = f"./TTT_data/Self_Certainty_Steplevel_best_of_{N}_second_{lambda_weight}_llama_1b_MATH500_3.json"
        with open(output_file, mode="w", encoding="utf-8") as file:
            json.dump({
                "results": table,
                "accuracy": accuracy
            }, file, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_file}")