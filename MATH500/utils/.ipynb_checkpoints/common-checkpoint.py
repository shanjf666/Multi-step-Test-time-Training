"""
Common utility functions for TTT methods
"""

import re
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from collections import Counter
from math_verify import parse

def parse_structured_steps(response_text):
    """
    解析结构化步骤（优先匹配 "## Step N:" 块），否则回退为句子/段落/行。
    """
    cleaned_text = re.sub(r'<\|eot_id\|>', '', response_text)
    cleaned_text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'<\|.*?\|>', '', cleaned_text)  # 泛化清除 <|...|>
    cleaned_text = cleaned_text.replace("<|end_of_text|>", "")
    cleaned_text = cleaned_text.replace("<|end_of_sentence|>", "")
    cleaned_text = cleaned_text.replace("</s>", "")
    cleaned_text = re.sub(r'<｜end▁of▁sentence｜>', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # 先匹配"## Step N: ..."
    step_pattern = r'##\s*Step\s*(\d+)\s*[:.]?\s*(.*?)(?=##\s*Step\s*\d+\s*[:.]?|$)'
    matches = re.findall(step_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
    if matches:
        steps = []
        for _, content in matches:
            content = content.strip()
            if content:
                steps.append(content)
        if steps:
            return steps

    # 退路1：按句子粗分
    sentences = re.split(r'[.!?]+', cleaned_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 2:
        return sentences

    # 退路2：按段落
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
    if len(paragraphs) >= 2:
        return paragraphs

    # 退路3：按行
    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
    return lines if lines else [cleaned_text]




def extract_model_answer(response_text):
    """
    从模型响应中提取最终数值/答案，适配若干常见格式。
    """
    cleaned_response = response_text.replace("<|begin_of_text|>", "").strip()
    cleaned_response = cleaned_response.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
    cleaned_response = cleaned_response.replace("<|eot_id|>", "").strip()
    cleaned_response = cleaned_response.replace("</s>", "").strip()
    cleaned_response = cleaned_response.replace("<|end_of_text|>", "").strip()
    cleaned_response = cleaned_response.replace("<|end_of_sentence|>", "").strip()
    cleaned_response = re.sub(r'<\|.*?\|>', '', cleaned_response)
    cleaned_response = re.sub(r'\s+', ' ', cleaned_response).strip()

    if not cleaned_response:
        return ""

    patterns = [
        r'Answer\s*[:\.\s]*([^\n]+)',
        r'####\s*([^\n]+)',
        r'\\boxed{(.+?)}',
        r'boxed{(.+?)}',
        r'The answer is\s*[:\.\s]*([^\.\n]+)',
        r'Final Answer\s*[:\.\s]*([^\.\n]+)',
        r'Therefore\s*,\s*([^\.\n]+)',
        r'答案[:：]\s*([^\n]+)',
        r'所以[:：]\s*([^\n]+)',
        r'final answer\s*[:\.\s]*([^\.\n]+)',
        r'(\d+(\.\d+)?)',
        r'<answer>\s*(.*?)\s*</answer>',
        r'<answer>(.*?)</answer>'
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned_response, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            number_match = re.search(r'([+-]?\d+(?:\.\d+)?)', answer)
            if number_match:
                return number_match.group(1)
            if answer:
                return answer

    lines = cleaned_response.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.lower().startswith(('question:', 'step', 'solution:', 'answer:', '问题:', '步骤:')):
            number_match = re.search(r'([+-]?\d+(?:\.\d+)?)$', line)
            if number_match:
                return number_match.group(1)

    number_match = re.search(r'([+-]?\d+(?:\.\d+)?)', cleaned_response)
    if number_match:
        return number_match.group(1)
    return ""

def is_correct_answer(predicted, true_answer):
    """
    判断预测答案与真值是否一致（针对MATH500数据集）
    """
    if not predicted or not true_answer:
        return False
    
    # 清理答案，移除多余空格
    pred_clean = re.sub(r'\s+', ' ', str(predicted)).strip()
    true_clean = re.sub(r'\s+', ' ', str(true_answer)).strip()
    
    # 直接比较（忽略空格和换行符）
    if pred_clean.replace(' ', '') == true_clean.replace(' ', ''):
        return True
    
    # 尝试数值比较（如果可能）
    try:
        pred_float = float(pred_clean.replace(',', ''))
        true_float = float(true_clean.replace(',', ''))
        return abs(pred_float - true_float) < 1e-6
    except ValueError:
        pass
    
    return False


def generate_with_transformers(model, tokenizer, prompt, device, temperature=0.7, max_tokens=512, num_return_sequences=1, output_hidden_states=False):
    """
    使用Transformers模型生成多个候选答案，返回 tokens 与 logits。
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    generate_kwargs = {
        "do_sample": True if temperature > 0 else False,
        "temperature": temperature,
        "top_p": 1.0,
        "max_new_tokens": max_tokens,
        "num_return_sequences": num_return_sequences,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "attention_mask": attention_mask,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    
    if output_hidden_states:
        generate_kwargs["output_hidden_states"] = True

    with torch.no_grad():
        outputs = model.generate(input_ids, **generate_kwargs)

    generated_sequences = []
    for i in range(num_return_sequences):
        sequence = outputs.sequences[i]
        response_ids = sequence[len(input_ids[0]):]  # 只保留响应部分

        # 重新前向，拿完整 logits（更稳妥）
        with torch.no_grad():
            full_outputs = model(input_ids=torch.unsqueeze(sequence, 0))
            logits = full_outputs.logits.squeeze(0)
            
        seq_data = {
            "tokens": response_ids,
            "logits": logits,
            "full_sequence": sequence,
            "prompt_ids": input_ids.squeeze(0)
        }
        
        if output_hidden_states and hasattr(outputs, 'hidden_states'):
            # 从完整的 outputs.hidden_states 中为当前序列 i 提取专属的 hidden_states
            single_sequence_hidden_states = tuple(
                tuple(
                    layer_tensor[i:i+1, :, :] for layer_tensor in step_hidden_states
                ) for step_hidden_states in outputs.hidden_states
            )
            seq_data["hidden_states"] = single_sequence_hidden_states
            
        generated_sequences.append(seq_data)
    return generated_sequences


def calculate_step_confidence_with_self_certainty(prompt_ids, response_ids, logits, tokenizer, lambda_weight=0.5):
    """
    使用完整概率分布计算步骤级置信度（对每步的token均值 + 生成对数概率融合）。
    返回：所有步骤置信度的平均值。
    """
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    steps = parse_structured_steps(response_text)

    prompt_length = len(prompt_ids)
    # -1 因为 logits[t] 预测的是下一个 token
    response_logits = logits[prompt_length - 1:-1, :]

    log_probs = F.log_softmax(response_logits, dim=-1)
    step_confidences = []
    token_index = 0

    for step_text in steps:
        if not step_text.strip():
            continue

        step_tokens = tokenizer.encode(step_text, add_special_tokens=False)
        step_length = len(step_tokens)
        if token_index + step_length > len(response_ids):
            step_length = len(response_ids) - token_index
        if step_length <= 0:
            continue

        token_confidences = []
        step_cumulative_logprob = 0.0

        for i in range(step_length):
            if token_index + i < len(response_ids) and token_index + i < len(log_probs):
                actual_token_id = response_ids[token_index + i].item()
                token_log_probs = log_probs[token_index + i]
                # 这里是"均值负对数概率"的一个近似指标，你的原始实现保留
                token_confidence = -torch.mean(token_log_probs).item()
                token_confidences.append(token_confidence)
                step_cumulative_logprob += token_log_probs[actual_token_id].item()

        if token_confidences:
            avg_token_confidence = sum(token_confidences) / len(token_confidences)
            step_probability = np.exp(step_cumulative_logprob)
            step_confidence = (avg_token_confidence ** (1 - lambda_weight)) * (step_probability ** lambda_weight)
            step_confidences.append(step_confidence)

        token_index += step_length

    final_confidence = 0.0
    for conf in step_confidences:
        final_confidence += conf
    if len(step_confidences) > 0:
        final_confidence /= len(step_confidences)
    else:
        final_confidence = 0.0
    return final_confidence


def calculate_step_confidence_with_self_eval(prompt_ids, response_ids, logits, tokenizer, lambda_weight=0.5, 
                                             eval_model=None, question="", device=None, steps=None):
    """
    根据公式计算步骤置信度（使用完整概率分布）
    
    Args:
        prompt_ids: 输入提示的token IDs
        response_ids: 生成响应的token IDs
        logits: 模型输出的logits
        tokenizer: 模型的分词器
        lambda_weight: 超参数，用于平衡自评分数和步骤概率的权重
        eval_model: 用于自评的模型
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
    
    # 计算响应部分的logits
    prompt_length = len(prompt_ids)
    response_logits = logits[prompt_length-1:-1, :]
    
    # 计算概率分布
    log_probs = F.log_softmax(response_logits, dim=-1)
    
    # 计算步骤自评分数
    if eval_model is not None and device is not None and question:
        step_evaluations = calculate_step_self_evaluation(eval_model, tokenizer, steps, question, device)
    else:
        step_evaluations = [0.5 for _ in steps]
    
    step_confidences = []
    token_index = 0
    
    # 为每个步骤提取概率分布并计算置信度
    for i, step_text in enumerate(steps):
        if not step_text.strip():
            continue
            
        step_tokens = tokenizer.encode(step_text, add_special_tokens=False)
        step_length = len(step_tokens)
        
        if token_index + step_length > len(response_ids):
            step_length = len(response_ids) - token_index
            
        if step_length <= 0:
            continue
            
        step_cumulative_logprob = 0.0
        
        for j in range(step_length):
            if token_index + j < len(response_ids) and token_index + j < len(log_probs):
                actual_token_id = response_ids[token_index + j].item()
                step_cumulative_logprob += log_probs[token_index + j][actual_token_id].item()
        
        step_evaluation_score = step_evaluations[i] if i < len(step_evaluations) else 0.5
        
        step_probability = np.exp(step_cumulative_logprob)
        step_confidence = step_evaluation_score**(1 - lambda_weight) * step_probability**lambda_weight
        step_confidences.append(step_confidence)
        
        token_index += step_length
    
    # 计算最终置信度
    final_confidence = 0.0
    if step_confidences:
        final_confidence = sum(step_confidences) / len(step_confidences)
    return final_confidence


def calculate_step_confidence_with_CoE_C(prompt_ids, response_ids, logits, hidden_states, tokenizer, lambda_weight=0.5):
    """
    使用完整概率分布计算步骤置信度
    """
    # 解码生成的文本
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    # 解析结构化步骤
    steps = parse_structured_steps(response_text)
    
    # 计算响应部分的logits
    prompt_length = len(prompt_ids)
    response_logits = logits[prompt_length-1:-1, :]
    
    # 计算概率分布
    log_probs = F.log_softmax(response_logits, dim=-1)
    
    # 存储每个步骤的置信度
    step_confidences = []
    
    # 当前处理的token索引
    token_index = 0
    
    # 为每个步骤提取概率分布并计算置信度
    for i, step_text in enumerate(steps):
        if not step_text.strip():
            continue
            
        # 计算当前步骤的token数量
        step_tokens = tokenizer.encode(step_text, add_special_tokens=False)
        step_length = len(step_tokens)
        
        # 确保不超出生成的token范围
        if token_index + step_length > len(response_ids):
            step_length = len(response_ids) - token_index
            
        if step_length <= 0:
            continue
            
        # 当前步骤的累积对数概率
        step_cumulative_logprob = 0.0
        
        # 遍历当前步骤中的每个token
        for j in range(step_length):
            if token_index + j < len(response_ids) and token_index + j < len(log_probs):
                actual_token_id = response_ids[token_index + j].item()
                step_cumulative_logprob += log_probs[token_index + j][actual_token_id].item()

        # 获取该步骤的CoE-C分数
        step_CoE_score = compute_CoE_C(token_index, step_length, hidden_states)
        
        # 结合自评分数和步骤概率
        step_probability = np.exp(step_cumulative_logprob)
        step_confidence = step_CoE_score**(1 - lambda_weight) * step_probability**lambda_weight
        step_confidences.append(step_confidence)
        
        # 更新token索引
        token_index += step_length
    
    # 所有步骤置信度的累加均值作为最终置信度
    final_confidence = 0.0
    for conf in step_confidences:
        final_confidence += conf
    final_confidence /= len(step_confidences) if len(step_confidences) > 0 else 1
    return final_confidence


def calculate_step_self_evaluation(model, tokenizer, steps, question, device):
    """
    对所有解题步骤进行自评
    
    Args:
        model: 评估模型
        tokenizer: 分词器
        steps: 步骤列表
        question: 原始问题
        device: 计算设备
        
    Returns:
        list: 每个步骤的自评分数列表
    """
    step_evaluations = []
    previous_steps = ""
    
    for i, step_text in enumerate(steps):
        evaluation_score = evaluate_step_correctness(
            model, tokenizer, step_text, question, device, previous_steps
        )
        step_evaluations.append(evaluation_score)
        previous_steps += f"## Step {i+1}: {step_text}\n"
    
    return step_evaluations


def evaluate_step_correctness(model, tokenizer, step_text, question, device, previous_steps=""):
    """
    使用模型评估单个解题步骤的正确性
    
    Args:
        model: 评估用的模型
        tokenizer: 分词器
        step_text: 要评估的步骤文本
        question: 原始问题
        device: 计算设备
        previous_steps: 之前的所有步骤
        
    Returns:
        float: 步骤正确性评分 (0-1)
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

    # 编码输入并生成响应
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]
    
    generate_kwargs = {
        "do_sample": False,
        "max_new_tokens": 10,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    
    with torch.no_grad():
        outputs = model.generate(input_ids, **generate_kwargs)
    
    # 解码生成的文本
    response_ids = outputs.sequences[0][len(input_ids[0]):]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    # 从logits中计算置信度
    if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
        def _check_eq(x, tokens):
            x = re.sub(r'[\(\)\s]', ' ', x).strip()
            return any(x == t for t in tokens) or any(x.lower() == t.lower() for t in tokens)
        
        w_tokens = ['B']  # 错误选项
        r_tokens = ['A']  # 正确选项
        
        for i, token_logits in enumerate(outputs.scores):
            probs = torch.softmax(token_logits, dim=-1)
            top_k = min(10, probs.size(-1))
            top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
            
            tlp = {}
            tokens_at_position = []
            for j in range(top_k):
                token_id = top_indices[0, j].item()
                prob = top_probs[0, j].item()
                token_text = tokenizer.decode([token_id]).strip()
                tlp[token_text] = prob
                tokens_at_position.append(token_text)
            
            generated_token_id = response_ids[i].item()
            generated_token_text = tokenizer.decode([generated_token_id]).strip()
            
            if any(_check_eq(token, w_tokens + r_tokens) for token in tokens_at_position):
                correct = sum(tlp.get(k, 0) for k in tlp if _check_eq(k, r_tokens))
                wrong = sum(tlp.get(k, 0) for k in tlp if _check_eq(k, w_tokens))
                confidence = correct
                return confidence
                
    return 0.5


def compute_CoE_C(token_index, step_length, hidden_states):
    """
    计算CoE-C分数（Change-over-Epoch based on Cosine similarity）
    """
    import math
    
    # 获取模型层数
    layer_num = len(hidden_states[1])
    hs_all_layer = []
    
    # 遍历每一层
    for j in range(layer_num):
        vectors_for_layer_j = []
        # 遍历指定的生成步骤
        for pos in range(token_index, token_index + step_length):
            # 处理不同位置的隐藏状态形状
            if pos == 0:
                # 第一步：张量形状为 (1, prompt_len, hidden_size)
                tensor = hidden_states[0][j]
                vector = tensor[0, -1] 
            else:
                # 后续步骤：张量形状为 (1, 1, hidden_size)
                tensor = hidden_states[pos][j]
                vector = tensor[0, 0]
            
            # 将PyTorch张量转换为NumPy数组
            vectors_for_layer_j.append(vector.float().cpu().numpy())
            
        # 计算该层在所有指定步骤上的平均隐藏状态
        all_pos_hs = np.array(vectors_for_layer_j)
        hs_all_layer.append(np.mean(all_pos_hs, axis=0))
        
    # 计算相邻层之间的幅度差异
    al_repdiff = np.array([hs_all_layer[i+1] - hs_all_layer[i] for i in range(layer_num - 1)])
    al_repdiff_norm = [np.linalg.norm(item, ord=2) for item in al_repdiff]
    
    # 计算相邻层之间的角度差异
    al_semdiff = []
    for i in range(layer_num - 1):
        a = hs_all_layer[i + 1]
        b = hs_all_layer[i]
        dot_product = np.dot(a, b)
        norm_a, norm_b = np.linalg.norm(a, ord=2), np.linalg.norm(b, ord=2)
        similarity = dot_product / (norm_a * norm_b + 1e-10)
        similarity = similarity if similarity <= 1 else 1
        similarity = similarity if similarity >= -1 else -1

        arccos_sim = math.acos(similarity)
        al_semdiff.append(arccos_sim)
    al_semdiff_norm = np.array(al_semdiff)
        
    # 将幅度和角度变化转换为二维坐标点
    x_list = np.array([al_repdiff_norm[i] * math.cos(al_semdiff_norm[i]) for i in range(len(al_semdiff_norm))])
    y_list = np.array([al_repdiff_norm[i] * math.sin(al_semdiff_norm[i]) for i in range(len(al_semdiff_norm))])
    al_combdiff_x_ave = np.mean(x_list)
    al_combdiff_y_ave = np.mean(y_list)

    # 返回最终CoE-C得分
    return math.sqrt(al_combdiff_x_ave ** 2 + al_combdiff_y_ave ** 2)

def clean_latex_format(text):
    """
    清理 LaTeX 格式中无用的符号与空白命令，保留数学表达式核心内容。
    
    功能：
      - 去掉 LaTeX 空白命令（如 \\quad, \\;, \\! 等）
      - 去掉 \\left, \\right, \\boxed{}, $ 等装饰符
      - 替换常见符号，如 \\pi -> pi
      - 清除多余空格
    
    Args:
        text (str): 包含 LaTeX 格式的字符串
        
    Returns:
        str: 清理后的字符串
    """
    if not text:
        return text

    cleaned_text = text

    # ===== 1️⃣ 去掉美元符号（$...$） =====
    cleaned_text = cleaned_text.replace("$", "")

    # ===== 2️⃣ 去掉 LaTeX 空白命令 =====
    latex_whitespaces = {
        r'\\,': ' ',
        r'\\:': ' ',
        r'\\;': ' ',
        r'\\!': '',
        r'\\enspace': ' ',
        r'\\quad': ' ',
        r'\\qquad': ' ',
        r'\\hspace{[^}]*}': ' ',
        r'\\vspace{[^}]*}': '',
        r'\\phantom{[^}]*}': '',
        r'\\hfill': ' ',
        r'\\space': ' ',
        r'\\ ': ' ',
    }
    for pattern, replacement in latex_whitespaces.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text)

    # ===== 3️⃣ 去掉常见无意义命令 =====
    useless_cmds = [
        r'\\left', r'\\right', r'\\big', r'\\Big', r'\\bigg', r'\\Bigg',
        r'\\text', r'\\mathrm', r'\\displaystyle'
    ]
    for cmd in useless_cmds:
        cleaned_text = re.sub(cmd, '', cleaned_text)

    # ===== 4️⃣ 特殊结构展开 =====
    cleaned_text = cleaned_text.replace(r'\boxed{', '')
    cleaned_text = cleaned_text.replace(r'\(', '(').replace(r'\)', ')')
    cleaned_text = cleaned_text.replace(r'\[', '[').replace(r'\]', ']')
    cleaned_text = cleaned_text.replace(r'\{', '{').replace(r'\}', '}')

    # ===== 5️⃣ 常见符号替换 =====
    cleaned_text = cleaned_text.replace(r'\pi', 'pi')
    cleaned_text = cleaned_text.replace(r'\theta', 'theta')
    cleaned_text = cleaned_text.replace(r'\sqrt', 'sqrt')

    # ===== 6️⃣ 去掉多余花括号、空格 =====
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text