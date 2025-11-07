"""
Entropy-based method implementation for MATH500 dataset
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
    calculate_step_confidence_with_entropy,
    clean_latex_format
)
from utils.key_step_extractor import summarize_key_steps_openai


def Entropy_Selection(dataset, config, model, tokenizer, device,
                      N=4, save_results=False, lambda_weight=0.5,
                      key_api_client=None,
                      key_api_model="qwen3-max-preview",
                      key_api_temp=0.2):
    """
    评估流程：多样本生成 -> 步骤置信度打分(基于熵) -> 选最优 -> 保存为 question/answer/gpt_response 三字段。
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
        # prompt = tokenizer.apply_chat_template(
        #     [{"role": "user", "content": f"{config.system_prompt}Q: {question}\n\nA:"}],
        #     tokenize=False, add_generation_prompt=True
        # )
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
                step_confidence = calculate_step_confidence_with_entropy(
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
            "max_confidence": step_confidence_scores[best_index],
            "correct": true_answer in clean_key_step_text[-30:] or is_correct_answer(model_answer, true_answer)
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
            print(f"最高置信度: {step_confidence_scores[best_index]:.4f}")
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

    # 写入JSON（每条为 {question, answer, gpt_response}）
    if save_results:
        os.makedirs("./TTT_data", exist_ok=True)
        output_file = f"./TTT_data/Entropy_best_of_{N}_lambda_{lambda_weight}_LLam1B_MATH500_3.json"
        with open(output_file, mode="w", encoding="utf-8") as file:
            json.dump({
                "results": table,
                "accuracy": accuracy
            }, file, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_file}")