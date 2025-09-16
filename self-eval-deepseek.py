"""
python self-eval-deepseek.py --subset_size 10 --lambda_weight 0.5 --model_path /root/autodl-tmp/data/models/modelscope_cache/models/deepseek-ai/deepseek-llm-7b-chat
python self-eval-deepseek.py --lambda_weight 1.0 --model_path /root/autodl-tmp/data/models/modelscope_cache/models/deepseek-ai/deepseek-llm-7b-chat
"""
# 禁用某些可能导致计算不一致的优化选项，确保结果的可重现性
import os
os.environ["TRITON_ALLOW_MMA"] = "0"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# 导入所需的各类库
import re  # 正则表达式处理
import json  # JSON文件处理
import time  # 时间计算
import torch  # PyTorch深度学习框架
import argparse  # 命令行参数解析
import numpy as np  # 数值计算
from tqdm import tqdm  # 进度条显示
from collections import Counter  # 计数器工具
from datasets import load_dataset  # Hugging Face数据集库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import math

# 配置类
class Config:
    def __init__(self):
        self.system_prompt = (
            "You are a helpful AI assistant good at solving math problems. "
            "Solve the following math problem step by step.\n\n"
            "Think carefully and provide a clear solution.\n\n"
            "For simple problems (2 steps or fewer):\n"
            "Provide a concise solution with minimal explanation.\n\n"
            "For complex problems (3 steps or more):\n"
            "Use this step-by-step format:\n\n"
            "## Step 1: [Brief description]\n"
            "[Explanation and calculations]\n\n"
            "## Step 2: [Brief description]\n"
            "[Explanation and calculations]\n\n"
            "...\n\n"
            "Always conclude with the final answer in this format:\n"
            "Therefore, the final answer is: $\\boxed{answer}$\n\n"
            "Where [answer] is just the final number or expression that solves the problem.\n"
            "Please follow this format strictly for all answers."
        )
        self.temperature = 0.1
        self.top_p = 1.0
        self.max_tokens = 512
        self.n = 4
        self.lambda_weight = 0.5
        self.custom_chat_template = None

#############################################################################################################
############################################# Self-Certainty Calculation ####################################
# Self-Certainty计算方法
#############################################################################################################

def parse_structured_steps(response_text):
    """
    解析结构化步骤的改进方法
    """
    # 清理文本
    cleaned_text = re.sub(r'<\|eot_id\|>', '', response_text)
    cleaned_text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'</|end_of_text|', '', cleaned_text)
    cleaned_text = re.sub(r'<｜end▁of▁sentence｜>', '', cleaned_text)
    cleaned_text = re.sub(r'<\|.*?\|>', '', cleaned_text)
    cleaned_text = re.sub(r'</s>', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # 首先尝试匹配 ## Step N: 格式的步骤
    step_pattern = r'##\s*Step\s*(\d+)\s*[:.]?\s*(.*?)(?=##\s*Step\s*\d+\s*[:.]?|$)'
    matches = re.findall(step_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
    
    if matches:
        steps = []
        for step_num, content in matches:
            content = content.strip()
            if content:
                steps.append(content)
        if steps:
            return steps
    
    # 如果没有找到 ## Step 格式，尝试其他分隔方式
    # 按照句子分割（以句号、问号、感叹号结尾的句子）
    sentences = re.split(r'[.!?]+', cleaned_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 最后的回退方案：按段落分割
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
    if len(paragraphs) >= 2:
        return paragraphs
    
    # 如果还是不行，按换行符分割
    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
    return lines if lines else [cleaned_text]

def evaluate_step_correctness(model, tokenizer, step_text, question, device, previous_steps=""):
    """
    使用模型自评单个步骤的正确性，参考extract_confidence_score的方式处理top_logprobs
    
    Args:
        model: 评估模型
        tokenizer: 分词器
        step_text: 要评估的步骤文本
        question: 原始问题
        device: 计算设备
        previous_steps: 之前的步骤（上下文）
        
    Returns:
        float: 步骤正确性评分 (0-1)，基于top_logprobs中正确选项的概率
    """
    # 构造评估提示
    prompt = f"""<|begin_of_text|>
You are an expert AI assistant that evaluates the correctness of problem-solving steps.

Task: Evaluate whether the following problem-solving step is correct.

Question: {question}

Previous steps:
{previous_steps}

Current step to evaluate:
{step_text}

Is this step correct? 
Options:
(A) Correct
(B) Incorrect

The answer is: """

    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    
    # 设置生成参数
    generate_kwargs = {
        "do_sample": False,  # 贪婪解码
        "max_new_tokens": 10,  # 生成足够长的文本以包含选项
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    
    # 生成评估结果，启用logits输出以获取概率信息
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            **generate_kwargs
        )
    
    # 解码生成的文本
    response_ids = outputs.sequences[0][len(input_ids[0]):]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    
    # 添加调试输出
    # print(f"\n--- 模型自评调试信息 ---")
    # print(f"评估提示: {prompt}")
    # print(f"模型回答: {response_text}")

    # 获取生成过程中的logits
    if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
        # 定义检查函数
        def _check_eq(x, tokens):
            x = re.sub(r'[\(\)\s]', ' ', x).strip()
            eq = False
            if any(x == t for t in tokens): 
                eq = True
            elif any(x.lower() == t.lower() for t in tokens): 
                eq = True
            return eq
        
        # 定义正确和错误的标记
        w_tokens = ['B']  # 错误选项
        r_tokens = ['A']  # 正确选项
        
        # 遍历生成的tokens和对应的top_logprobs
        for i, token_logits in enumerate(outputs.scores):
            probs = torch.softmax(token_logits, dim=-1)
            
            # 获取top tokens
            top_k = min(10, probs.size(-1))
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            
            # 构建top_logprobs格式的数据
            tlp = {}
            tokens_at_position = []
            for j in range(top_k):
                token_id = top_indices[0, j].item()
                prob = top_probs[0, j].item()
                token_text = tokenizer.decode([token_id]).strip()
                tlp[token_text] = prob
                tokens_at_position.append(token_text)
            
            # 检查当前位置是否包含A/B选项信息
            generated_token_id = response_ids[i].item()
            generated_token_text = tokenizer.decode([generated_token_id]).strip()
            
            if any(_check_eq(token, w_tokens + r_tokens) for token in tokens_at_position):
                # 计算正确选项的总概率
                correct = sum(tlp.get(k, 0) for k in tlp if _check_eq(k, r_tokens))
                # 计算错误选项的总概率
                wrong = sum(tlp.get(k, 0) for k in tlp if _check_eq(k, w_tokens))
                
                # 返回正确选项的概率作为置信度
                confidence = correct
                return confidence
    print(f"警告: 未能识别明确的A/B选项选择，返回默认置信度0.5")
    return 0.5

def calculate_step_self_evaluation(model, tokenizer, steps, question, device):
    """
    对所有步骤进行自评
    
    Args:
        model: 评估模型
        tokenizer: 分词器
        steps: 步骤列表
        question: 原始问题
        device: 计算设备
        
    Returns:
        list: 每个步骤的自评分数
    """
    step_evaluations = []
    previous_steps = ""
    
    for i, step_text in enumerate(steps):
        # 获取步骤自评分数
        evaluation_score = evaluate_step_correctness(
            model, tokenizer, step_text, question, device, previous_steps
        )
        step_evaluations.append(evaluation_score)
        
        # 更新previous_steps用于下一个步骤的评估
        previous_steps += f"## Step {i+1}: {step_text}\n"
    
    return step_evaluations

def calculate_step_confidence_with_full_probs(prompt_ids, response_ids, logits, tokenizer, lambda_weight=0.5, 
                                            model=None, question="", device=None, steps=None):
    """
    根据公式计算步骤置信度（使用完整概率分布）
    使用模型自评分数替换token置信度
    
    Args:
        prompt_ids: 输入提示的token IDs
        response_ids: 生成响应的token IDs
        logits: 模型输出的logits (seq_len, vocab_size)
        tokenizer: 模型的分词器
        lambda_weight: 超参数，用于平衡自评分数和步骤概率的权重
        model: 用于自评的模型
        question: 原始问题
        device: 计算设备
        steps: 解析后的步骤列表
        
    Returns:
        float: 步骤级置信度
    """
    # 如果没有提供steps，解析生成的文本
    if steps is None:
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        steps = parse_structured_steps(response_text)
    
    # 计算响应部分的logits（跳过输入部分）
    prompt_length = len(prompt_ids)
    response_logits = logits[prompt_length-1:-1, :]  # -1因为下一个token预测
    
    # 计算概率分布
    log_probs = F.log_softmax(response_logits, dim=-1)
    
    # 获取步骤自评分数
    if model is not None and device is not None and question:
        step_evaluations = calculate_step_self_evaluation(model, tokenizer, steps, question, device)
    else:
        # 如果没有提供自评模型，使用默认的token置信度
        step_evaluations = []
        for _ in steps:
            step_evaluations.append(0.5)  # 默认评分
    
    # 初始化存储每个步骤信息的列表
    step_confidences = []  # 存储每个步骤的置信度
    
    # 当前处理的token索引
    token_index = 0
    
    # 为每个步骤提取概率分布并计算置信度
    for i, step_text in enumerate(steps):
        # 跳过空步骤
        if not step_text.strip():
            continue
            
        # 计算当前步骤的token数量
        step_tokens = tokenizer.encode(step_text, add_special_tokens=False)
        step_length = len(step_tokens)
        
        # 确保不超出生成的token范围
        if token_index + step_length > len(response_ids):
            step_length = len(response_ids) - token_index
            
        # 如果步骤长度无效，则跳过
        if step_length <= 0:
            continue
            
        # 当前步骤的累积对数概率
        step_cumulative_logprob = 0.0
        
        # 遍历当前步骤中的每个token
        for j in range(step_length):
            # 确保token索引在有效范围内
            if token_index + j < len(response_ids) and token_index + j < len(log_probs):
                # 获取实际生成token的ID
                actual_token_id = response_ids[token_index + j].item()
                # 获取实际生成的token的logprob
                step_cumulative_logprob += log_probs[token_index + j][actual_token_id].item()
                
        # step_cumulative_logprob /= step_length

        # 获取该步骤的自评分数
        step_evaluation_score = step_evaluations[i] if i < len(step_evaluations) else 0.5
        
        # 将自评分数考虑进来，使用lambda进行加权
        step_probability = np.exp(step_cumulative_logprob)
        # 结合自评分数和步骤概率
        step_confidence = step_evaluation_score**(1 - lambda_weight) * step_probability**lambda_weight
        step_confidences.append(step_confidence)
        
        # 更新token索引
        token_index += step_length
    
    # 所有步骤置信度的平均作为最终置信度
    final_confidence = 1.0
    if step_confidences:
        for conf in step_confidences:
            final_confidence *= conf
        # final_confidence /= len(step_confidences)
    return final_confidence
    
#############################################################################################################
############################################# Answer Extraction #############################################
# 答案提取方法
#############################################################################################################
def extract_model_answer(response_text):
    """
    从模型响应中提取答案，适配不同模型格式
    """
    # 清理响应文本，移除特殊标记
    cleaned_response = response_text.replace("<|begin_of_text|>", "").strip()
    cleaned_response = cleaned_response.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
    cleaned_response = cleaned_response.replace("<|eot_id|>", "").strip()
    cleaned_response = cleaned_response.replace("</s>", "").strip()
    cleaned_response = cleaned_response.replace("<|end_of_text|>", "").strip()  # DeepSeek 特殊标记
    cleaned_response = cleaned_response.replace("<|end_of_sentence|>", "").strip() 
    
    # 如果清理后为空，返回空字符串
    if not cleaned_response:
        return ""
    
    # 多种答案格式的匹配模式，针对数学问题优化
    patterns = [
        r'Answer\s*[:\.\s]*([^\n]+)',
        r'####\s*([^\n]+)',
        r'\\boxed{(.+?)}',
        r'boxed{(.+?)}',
        r'The answer is\s*[:\.\s]*([^\.\n]+)',
        r'Final Answer\s*[:\.\s]*([^\.\n]+)',
        r'Therefore\s*,\s*([^\.\n]+)',
        r'答案[:：]\s*([^\n]+)',  # 中文答案
        r'所以[:：]\s*([^\n]+)',   # 中文所以
        r'final answer\s*[:\.\s]*([^\.\n]+)',  # 处理 "final answer" 的情况
        r'(\d+(\.\d+)?)'  # 提取数字
    ]
    
    # 尝试每种模式
    for pattern in patterns:
        match = re.search(pattern, cleaned_response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # 进一步清理答案，提取数字
            number_match = re.search(r'([+-]?\d+(?:\.\d+)?)', answer)
            if number_match:
                return number_match.group(1)
            if answer:  # 如果匹配到但没有数字，返回原始答案
                return answer
    
    # 如果没有找到标准格式，尝试提取最后一行的数字
    lines = cleaned_response.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.lower().startswith(('question:', 'step', 'solution:', 'answer:', '问题:', '步骤:')):
            # 查找行中的数字
            number_match = re.search(r'([+-]?\d+(?:\.\d+)?)$', line)
            if number_match:
                return number_match.group(1)
    
    # 最后的后备方案：在整个文本中查找数字
    number_match = re.search(r'([+-]?\d+(?:\.\d+)?)', cleaned_response)
    if number_match:
        return number_match.group(1)
    
    return ""  # 如果什么都没找到，返回空字符串

def is_correct_answer(predicted, true_answer):
    """
    检查预测答案是否正确
    
    Args:
        predicted: 模型预测的答案
        true_answer: 真实答案
    
    Returns:
        bool: 答案是否正确
    """
    if not predicted or not true_answer:
        return False
    
    try:
        # 尝试数值比较
        pred_float = float(str(predicted).replace(',', ''))
        true_float = float(str(true_answer).replace(',', ''))
        return abs(pred_float - true_float) < 1e-6
    except ValueError:
        # 如果无法转换为数值，使用字符串比较
        return (str(true_answer).strip() == str(predicted).strip() or 
                str(true_answer).replace(" ", "") == str(predicted).replace(" ", ""))

#############################################################################################################
############################################# Generation ####################################################
# 使用Transformers生成响应
#############################################################################################################

def generate_with_transformers(model, tokenizer, prompt, device, temperature=0.7, max_tokens=512, num_return_sequences=1):
    """
    使用Transformers模型生成多个候选答案
    
    Args:
        model: Transformers模型
        tokenizer: 分词器
        prompt: 输入提示
        device: 计算设备
        temperature: 温度参数
        max_tokens: 最大生成token数
        num_return_sequences: 生成序列数量
    
    Returns:
        list: 生成的响应列表，每个包含tokens和logits
    """
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # 设置生成参数
    generate_kwargs = {
        "do_sample": True if temperature > 0 else False,
        "temperature": temperature,
        "top_p": 1.0,
        "max_new_tokens": max_tokens,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "attention_mask": attention_mask,
    }
    
    # 生成多个候选答案
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            output_scores=True,
            return_dict_in_generate=True,
            **generate_kwargs
        )
    
    # 提取生成的序列和对应的logits
    generated_sequences = []
    for i in range(num_return_sequences):
        # 获取生成的token IDs
        sequence = outputs.sequences[i]
        response_ids = sequence[len(input_ids[0]):]  # 只保留响应部分
        
        # 重新运行模型以获取完整logits
        with torch.no_grad():
            full_outputs = model(input_ids=torch.unsqueeze(sequence, 0))
            logits = full_outputs.logits.squeeze(0)
        
        generated_sequences.append({
            "tokens": response_ids,
            "logits": logits,
            "full_sequence": sequence,
            "prompt_ids": input_ids.squeeze(0)
        })
    
    return generated_sequences

#############################################################################################################
############################################# Main Selection ################################################
# 主选择方法
#############################################################################################################

def Self_Certainty_Selection(dataset, config: Config, model, tokenizer, device, N=4, save_results=False, lambda_weight=0.5):
    """
    基于步骤的置信度计算并选择最佳答案的函数（使用完整概率分布）
    
    Args:
        dataset: 测试数据集
        config: 配置对象
        model: Transformers模型
        tokenizer: 分词器
        device: 计算设备
        N: 为每个问题生成的解决方案数量
        save_results: 是否保存结果到JSON文件
        lambda_weight: 超参数，用于平衡token置信度和步骤概率的权重
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
        
        # # 构建提示格式（适配Llama 3.2）
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

        # 计算每个响应的步骤级置信度分数
        step_confidence_scores = []
        valid_candidates = []
        candidate_answers = []
        
        # 在生成候选答案后，修改调用 calculate_step_confidence_with_full_probs 的部分
        for candidate in candidates:
            try:
                # 使用新的置信度计算方法（包含步骤概率和lambda权重）
                step_confidence = calculate_step_confidence_with_full_probs(
                    candidate["prompt_ids"], 
                    candidate["tokens"], 
                    candidate["logits"], 
                    tokenizer,
                    lambda_weight=lambda_weight,
                    model=model,  # 传递模型用于自评
                    question=question,  # 传递问题
                    device=device,  # 传递设备
                    steps=parse_structured_steps(tokenizer.decode(candidate["tokens"], skip_special_tokens=True))  # 传递解析的步骤
                )
                step_confidence_scores.append(step_confidence)
                valid_candidates.append(candidate)
                
                # 提取答案用于调试
                response_text = tokenizer.decode(candidate["tokens"], skip_special_tokens=False)
                extracted_answer = extract_model_answer(response_text)
                candidate_answers.append(extracted_answer)
                
            except Exception as e:
                print(f"Error calculating step confidence for candidate: {e}")
                step_confidence_scores.append(float('-inf'))  # 给错误的候选者最低分数
                valid_candidates.append(candidate)
                candidate_answers.append("")

        if not step_confidence_scores or all(score == float('-inf') for score in step_confidence_scores):
            print(f"No valid candidates for sample {i}")
            continue
            
        # 选择步骤级置信度最高的响应
        best_index = step_confidence_scores.index(max(step_confidence_scores))
        best_candidate = valid_candidates[best_index]
        
        # 解码最佳响应
        response_text = tokenizer.decode(best_candidate["tokens"], skip_special_tokens=False)
        
        # 提取模型答案
        model_answer = extract_model_answer(response_text)
        
        # 检查答案是否正确
        n_samples += 1
        if is_correct_answer(model_answer, true_answer):
            n_true_ans += 1

        # 清理控制标记（关键！）
        cleaned_text = re.sub(r'<\|eot_id\|>', '', response_text)
        cleaned_text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', cleaned_text, flags=re.DOTALL)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        cleaned_text = re.sub(r'<｜end▁of▁sentence｜>', '', cleaned_text)
        cleaned_text = re.sub(r'</|end_of_text|', '', cleaned_text)
        
        # 保存结果
        table.append({
            "ID": index+1, 
            "model_input": question, 
            "output": [cleaned_text],
            "model_answer": model_answer,  
            "true_answer": true_answer,
            "confidence": step_confidence_scores[best_index],
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
            print(f"清理后文本: {cleaned_text}")
            print(f"最高置信度: {step_confidence_scores[best_index]:.4f}")
            print(f"是否正确: {is_correct_answer(model_answer, true_answer)}")
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
        output_file = f"./TTT_data/Best_of_{N}_Transformers_Step_Eval_lambda_{lambda_weight}_deepseek_7b_multiply.json"
        with open(output_file, mode="w", encoding="utf-8") as file:
            json.dump(table, file, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_file}")

#############################################################################################################
################################################### Main ####################################################
# 程序入口点
#############################################################################################################

if __name__ == "__main__":
    # 创建配置对象
    config = Config()
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="基于步骤置信度的评估方法（使用Transformers）")
    
    # 添加命令行参数
    parser.add_argument("--n_repetitive_sampling", default=4, type=int, help="为每个问题生成的解决方案数量")
    parser.add_argument("--temperature", default=0.1, type=float, help="生成时的温度参数，控制随机性")
    parser.add_argument("--top_p", default=1.0, type=float, help="Top-p采样参数")
    parser.add_argument("--model_path", default="meta-llama/Llama-3.2-1B-Instruct", help="基础模型路径")
    parser.add_argument("--save_to_json", default=True, action="store_true", help="是否将结果保存到JSON文件")
    parser.add_argument("--dataset_repo_name", default="openai/gsm8k", help="数据集仓库名称")
    parser.add_argument("--max_tokens", default=512, type=int, help="最大生成标记数")
    parser.add_argument("--subset_size", default=None, type=int, help="测试子集大小（用于快速测试）")
    parser.add_argument("--lambda_weight", default=0.5, type=float, 
                       help="置信度计算中的lambda权重参数，用于平衡token置信度和步骤概率")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 将解析的参数赋值给配置对象
    config.n = int(args.n_repetitive_sampling)
    config.temperature = float(args.temperature)
    config.top_p = float(args.top_p)
    config.max_tokens = int(args.max_tokens)
    config.lambda_weight = float(args.lambda_weight)

    # 设置设备
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
    
    # 添加padding token如果不存在
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))
    
    model.eval()
    print("语言模型加载完成!")

    # 加载测试数据集
    print("正在加载数据集...")
    dataset = load_dataset(args.dataset_repo_name, 'main', split="test")
    
    # 如果指定了子集大小，则只使用部分数据进行测试
    if args.subset_size:
        dataset = dataset.select(range(min(args.subset_size, len(dataset))))
        
    print(f"数据集加载完成，共{len(dataset)}个样本")

    # 执行基于步骤的置信度评估
    print("开始执行基于步骤的置信度评估...")
    Self_Certainty_Selection(
        dataset, 
        config=config, 
        model=model,
        tokenizer=tokenizer,
        device=device,
        N=config.n, 
        save_results=args.save_to_json,
        lambda_weight=config.lambda_weight
    )

    # 打印所有参数及其值
    print("########################################################################################")
    print("实验配置参数:")
    print("########################################################################################")
    for arg, value in vars(args).items():
        print(f"{arg:>25} ===> {value}")
    print("########################################################################################")