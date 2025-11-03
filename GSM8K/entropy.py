"""
export OPENAI_API_KEY='sk-szeemiuaxtqymzkkpaezgadntxpajxwyhrzofzsodywqngqz'
python entropy_avglogp.py --lambda_weight 1.0 --temperature 0.7 --model_path Qwen/Qwen2.5-Math-7B-Instruct --subset_size 100 --n_repetitive_sampling 1 --lambda_weight 0.5 --model_path /root/autodl-tmp/data/models/modelscope_cache/models/deepseek-ai/deepseek-llm-7b-chat
python entropy_avglogp.py --subset_size 100 --lambda_weight 0.5 --model_path /root/autodl-tmp/Llama-3.2-1B-Instruct --n_repetitive_sampling 4 --max_tokens 512 --key_step_api_model Qwen/Qwen3-Next-80B-A3B-Thinking
python entropy_avglogp.py --lambda_weight 1 --model_path /root/autodl-tmp/data/models/modelscope_cache/models/deepseek-ai/deepseek-llm-7b-chat
"""
# 禁用某些可能导致计算不一致的优化选项，确保结果的可重现性
import os
os.environ["TRITON_ALLOW_MMA"] = "0"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
from openai import OpenAI  

# 配置类
class Config:
    def __init__(self):
        self.system_prompt = (
            """
                Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
                Answer: ## Step1: There are 15 trees originally. ## Step2: Then there were 21 trees after some more were planted. ## Step3: So there must have been 21 - 15 = 6. The answer is 6.
                
                
                Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
                Answer: ## Step1: There are originally 3 cars. ## Step2: 2 more cars arrive. ## Step3: 3 + 2 = 5. The answer is 5.
                
                
                Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
                Answer: ## Step1: Originally, Leah had 32 chocolates. ## Step2: Her sister had 42. ## Step3: So in total they had 32 + 42 = 74. ## Step4: After eating 35, they had 74 - 35 = 39. The answer is 39.
                
                
                Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
                Answer: ## Step1: Jason started with 20 lollipops. ## Step2: Then he had 12 after giving some to Denny. ## Step3: So he gave Denny 20 - 12 = 8. The answer is 8.
                
                
                Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
                Answer: ## Step1: Shawn started with 5 toys. ## Step2: If he got 2 toys each from his mom and dad, then that is 4 more toys. ## Step3: 5 + 4 = 9. The answer is 9.
                
                
                Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
                Answer: ## Step1: There were originally 9 computers. ## Step2: For each of 4 days, 5 more computers were added. ## Step3: So 5 * 4 = 20 computers were added. ## Step4: 9 + 20 is 29. The answer is 29.
                
                
                Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
                Answer: ## Step1: Michael started with 58 golf balls. ## Step2: After losing 23 on tuesday, he had 58 - 23 = 35. ## Step3: After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
                
                
                Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
                Answer: ## Step1: Olivia had 23 dollars. ## Step2: 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. ## Step3: So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.
            """
        )
        self.temperature = 0.1
        self.top_p = 1.0
        self.max_tokens = 512
        self.n = 4
        self.custom_chat_template = None
        self.lambda_weight = 0.7

#############################################################################################################
############################################# Self-Certainty Calculation ####################################
# Self-Certainty计算方法
#############################################################################################################

def parse_structured_steps(response_text):
    """
    从模型响应中解析结构化步骤。
    优先匹配：
      1. "## Step N:" 结构（最高优先级）
      2. "Step N:" 结构（次优先级）
    若未匹配到结构化步骤，则回退为句子、段落或行拆分。
    """
    # === 1. 清洗无关标记 ===
    cleaned_text = re.sub(r'<\|eot_id\|>', '', response_text)
    cleaned_text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'<\|.*?\|>', '', cleaned_text)  # 泛化清除 <|...|>
    cleaned_text = cleaned_text.replace("<|end_of_text|>", "")
    cleaned_text = cleaned_text.replace("<|end_of_sentence|>", "")
    cleaned_text = cleaned_text.replace("</s>", "")
    cleaned_text = re.sub(r'<｜end▁of▁sentence｜>', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # === 2. 匹配 "## Step N:" 结构 ===
    step_pattern_h2 = r'##\s*Step\s*(\d+)\s*[:.]?\s*(.*?)(?=##\s*Step\s*\d+\s*[:.]?|$)'
    matches_h2 = re.findall(step_pattern_h2, cleaned_text, re.DOTALL | re.IGNORECASE)
    if matches_h2:
        
        steps = [content.strip() for _, content in matches_h2 if content.strip()]
        if steps:
            return steps

    # === 3. 匹配 "Step N:" 结构 ===
    step_pattern_plain = r'\bStep\s*(\d+)\s*[:.]?\s*(.*?)(?=\bStep\s*\d+\s*[:.]?|$)'
    matches_plain = re.findall(step_pattern_plain, cleaned_text, re.DOTALL | re.IGNORECASE)
    if matches_plain:
        steps = [content.strip() for _, content in matches_plain if content.strip()]
        if steps:
            return steps

    # === 4. 回退方案：句子 / 段落 / 行 ===
    # 句子分割（粗略）
    sentences = re.split(r'(?<=[.!?。！？])\s+', cleaned_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 2:
        return sentences

    # 段落分割
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
    if len(paragraphs) >= 2:
        return paragraphs

    # 行分割
    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
    return lines if lines else [cleaned_text]


def find_subsequence(haystack, needle):
    """
    在 haystack (list of ints) 中查找 needle (list of ints) 的起始索引。
    如果未找到，返回 None。
    """
    if not needle:
        return None
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return None


def calculate_entropy_based_confidence(prompt_ids, response_ids, logits, tokenizer, lambda_weight=0.5):
    """
    使用熵值计算步骤级置信度，并添加基于 log prob 的答案置信度。
    最终分数：lambda * avg_step_conf + (1 - lambda) * answer_conf
    """
    
    # 解码输出文本并解析结构化步骤
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    steps = parse_structured_steps(response_text)
    
    # 提取响应部分的 logits（因为 logits[t] 预测的是下一个 token）
    prompt_length = len(prompt_ids)
    response_logits = logits[prompt_length - 1:-1, :]
    
    # 数值稳定的 softmax 计算（float32 + 归一化）
    response_logits = response_logits.float()
    response_logits = response_logits - response_logits.max(dim=-1, keepdim=True).values
    probs = torch.nn.functional.softmax(response_logits, dim=-1)
    
    # 计算每个token的熵
    log_probs_dist = torch.log(probs + 1e-20)
    entropy = -(probs * log_probs_dist).sum(dim=-1)
    
    # 最大熵（用于归一化）
    vocab_size = logits.shape[-1]
    max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float))
    
    # 分步骤计算置信度
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
    
        # 获取当前步骤的熵值
        step_entropies = []
        for i in range(step_length):
            full_index = token_index + i
            if full_index < len(entropy):
                ent_val = entropy[full_index].item()
                step_entropies.append(ent_val)
    
        # 平均并归一化
        if step_entropies:
            avg_entropy = sum(step_entropies) / len(step_entropies)
            normalized_entropy = avg_entropy / max_entropy.item()
            step_confidence = 1.0 - normalized_entropy  # 熵越低，置信度越高
            step_confidences.append(step_confidence)
    
        token_index += step_length
    
    # 步骤置信度的平均（用于融合）
    # if step_confidences:
    #     avg_step_conf = sum(step_confidences) / len(step_confidences)
    # else:
    #     avg_step_conf = 0.0
    min_step_conf = min(step_confidences)
    
    # 新增：计算答案置信度基于 log prob
    model_answer = extract_model_answer(response_text)
    if not model_answer:
        answer_confidence = 0.0  # 如果无答案，置信度为0
    else:
        answer_tokens = tokenizer.encode(model_answer, add_special_tokens=False)
        if not answer_tokens:
            answer_confidence = 0.0
        else:
            # 在 response_ids 中定位答案子序列
            answer_start = find_subsequence(response_ids.tolist(), answer_tokens)
            if answer_start is None:
                # Fallback: 假设答案在末尾，取最后 len(answer_tokens) 个
                answer_start = max(0, len(response_ids) - len(answer_tokens))
            
            answer_end = answer_start + len(answer_tokens)
            
            # 确保索引有效
            if answer_start >= len(probs) or answer_end > len(probs):
                answer_confidence = 0.0
            else:
                # 计算每个答案token的 log prob (实际选择的token)
                answer_log_probs = []
                for i in range(answer_start, answer_end):
                    actual_token = response_ids[i]
                    prob = probs[i, actual_token].item()
                    log_prob = np.log(prob + 1e-20)  # 使用 np.log 以防 torch.log 问题
                    answer_log_probs.append(log_prob)
                
                if answer_log_probs:
                    avg_log_prob = sum(answer_log_probs) / len(answer_log_probs)
                    answer_confidence = np.exp(avg_log_prob)  # exp(avg_log_prob)
                else:
                    answer_confidence = 0.0
    
    # 融合：加权平均
    # final_confidence = lambda_weight * avg_step_conf + (1 - lambda_weight) * answer_confidence
    final_confidence = lambda_weight * min_step_conf + (1 - lambda_weight) * answer_confidence
    return final_confidence



# ============================================================================
#                                Answer Extraction
# ============================================================================

_SOLUTION_CLIP_CHARS = 100  # 限制匹配范围，提升性能

def extract_model_answer(response_text):
    """
    从模型响应中提取数值型最终答案，适配常见推理输出格式。
    优先提取最后出现的有效数字（整数/小数/负数），用于数学推理任务。
    """

    # === 1. 清洗生成标记与特殊符号 ===
    cleaned = re.sub(
        r"<\|.*?\|>|</s>|</?answer>|</?s>|<s>|<eos>|<bos>|<pad>",
        "", response_text, flags=re.IGNORECASE
    ).strip()
    cleaned = re.sub(r'\s+', ' ', cleaned)

    if not cleaned:
        return ""

    # === 2. 限制匹配范围（加速处理） ===
    if len(cleaned) > _SOLUTION_CLIP_CHARS:
        cleaned = cleaned[-_SOLUTION_CLIP_CHARS:]

    # === 3. 优先匹配显式格式（####, Answer:, boxed{}, 等） ===
    patterns = [
        r'####\s*([\-0-9\.,]+)',
        r'Answer\s*[:\.\s]*([\-0-9\.,]+)',
        r'Final Answer\s*[:\.\s]*([\-0-9\.,]+)',
        r'The answer is\s*[:\.\s]*([\-0-9\.,]+)',
        r'\\boxed{([\-0-9\.,]+)}',
        r'boxed{([\-0-9\.,]+)}',
        r'最终答案[:：]\s*([\-0-9\.,]+)',
        r'答案[:：]\s*([\-0-9\.,]+)',
        r'所以[:：]\s*([\-0-9\.,]+)',
        r'Therefore\s*,?\s*([\-0-9\.,]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            ans = match.group(1).replace(",", "").replace("$", "").strip()
            if re.match(r'^[+-]?\d+(\.\d+)?$', ans):
                return ans

    # === 4. 回退策略：提取最后一个数值 ===
    all_numbers = re.findall(r'([+-]?\d+(?:\.\d+)?)', cleaned)
    if all_numbers:
        return all_numbers[-1]

    return ""


def is_correct_answer(predicted, true_answer):
    """
    判断预测答案与真值是否一致（优先数值比较，否则字符串比较）
    """
    if not predicted or not true_answer:
        return False
    try:
        pred_float = float(str(predicted).replace(',', ''))
        true_float = float(str(true_answer).replace(',', ''))
        return abs(pred_float - true_float) < 1e-6
    except ValueError:
        return (str(true_answer).strip() == str(predicted).strip() or
                str(true_answer).replace(" ", "") == str(predicted).replace(" ", ""))


# ============================================================================
#                                Key Step Extraction (OpenAI only)
# ============================================================================

KEY_STEP_PROMPT_TEMPLATE = """You are a reasoning step extraction model for mathematical problems.

Your goal: extract only symbolic or numeric computation steps that directly lead to the answer.

Rules:
1. Remove all narrative or explanatory phrases (e.g., "so", "therefore", "she has").
2. Keep only equations, arithmetic operations, or proportional relationships.
3. For each computation, list equivalent forms: symbolic (e.g., 3×5), decimal (15.0), and verbal (e.g., "3 times 5") and latex (e.g., $3 \\times 5$).
4. Merge trivial transformations (e.g., “= 15” and “equals 15”) into one step.
5. Omit the final answer step.
6. Do not correct any mistakes in the reasoning—extract what is actually used, even if flawed
7. Output in the following JSON format:
{{
  "key_steps": [
    ["16 - 3 = 13"],
    ["13 - 4 = 9"],
    ["9 × 2 = 18", "9 * 2","9 times 2"]
  ]
}}
Now extract the key reasoning steps from the reasoning below:

{reasoning}
"""

def _clean_for_summary(text: str) -> str:
    # 清理模型输出中的控制符与模板残留
    cleaned = re.sub(r'<\|eot_id\|>', '', text)
    cleaned = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<\|.*?\|>', '', cleaned)  # 清除 <|...|>
    cleaned = cleaned.replace("<|end_of_text|>", "")
    cleaned = cleaned.replace("<|end_of_sentence|>", "")
    cleaned = cleaned.replace("</s>", "")
    cleaned = re.sub(r'<｜end▁of▁sentence｜>', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def summarize_key_steps_openai(client: "OpenAI",
                               model: str,
                               reasoning_text: str,
                               temperature: float = 0.2) -> str:
    """
    使用 OpenAI 兼容接口抽取关键步骤（不做本地启发式回退）。
    """
    prompt = KEY_STEP_PROMPT_TEMPLATE.format(reasoning=_clean_for_summary(reasoning_text))
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert reasoning summarizer."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    content = (resp.choices[0].message.content or "").strip() if resp and resp.choices else ""
    return content


# ============================================================================
#                                Generation
# ============================================================================

def generate_with_transformers(model, tokenizer, prompt, device, temperature=0.7, max_tokens=512, num_return_sequences=1):
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

        generated_sequences.append({
            "tokens": response_ids,
            "logits": logits,
            "full_sequence": sequence,
            "prompt_ids": input_ids.squeeze(0)
        })
    return generated_sequences


# ============================================================================
#                                Main Selection
# ============================================================================
def clean_latex_format(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""

    cleaned_text = text.strip()

    # ===== 1️⃣ 移除美元符号和转义美元符号 =====
    cleaned_text = cleaned_text.replace("$", "").replace("\\$", "$")

    # ===== 2️⃣ 移除 LaTeX 空白命令 =====
    latex_whitespaces = {
        r'\\,': '', r'\\:': '', r'\\;': '', r'\\!': '',
        r'\\enspace': '', r'\\quad': '', r'\\qquad': '',
        r'\\hspace{[^}]*}': '', r'\\vspace{[^}]*}': '',
        r'\\phantom{[^}]*}': '', r'\\hfill': '', r'\\space': '',
        r'\\ ': '', r'\\mspace{[^}]*}': '', r'\\kern{[^}]*}': '',
    }
    for pattern, replacement in latex_whitespaces.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text)

    # ===== 3️⃣ 移除装饰性命令 =====
    useless_cmds = [
        r'\\left', r'\\right', r'\\big', r'\\Big', r'\\bigg', r'\\Bigg',
        r'\\text\s*', r'\\mathrm', r'\\displaystyle', r'\\rm', r'\\it',
        r'\\bf', r'\\cal', r'\\scriptstyle', r'\\scriptscriptstyle'
    ]
    for cmd in useless_cmds:
        cleaned_text = re.sub(cmd, '', cleaned_text)

    # ===== 4️⃣ 展开特殊结构 =====
    cleaned_text = cleaned_text.replace(r'\(', '(').replace(r'\)', ')')
    cleaned_text = cleaned_text.replace(r'\[', '[').replace(r'\]', ']')
    cleaned_text = cleaned_text.replace(r'\{', '{').replace(r'\}', '}')

    # ===== 5️⃣ 扩展符号替换（SymPy 兼容） =====
    symbol_replacements = {
        r'\\pi': 'pi', r'\\theta': 'theta', r'\\sqrt': 'sqrt',
        r'\\frac{([^}]*?)}({[^}]*?})': r'\1/\2',  # \frac{a}{b} -> a/b
        r'\\times': '*', r'\\div': '/', r'\\infty': 'oo',
        r'\\alpha': 'alpha', r'\\beta': 'beta', r'\\gamma': 'gamma',
        r'\\delta': 'delta', r'\\sum': 'sum', r'\\int': 'integrate',
        r'\\cdot': '*', r'\\pm': '+-', r'\\mp': '-+'
    }
    for pattern, replacement in symbol_replacements.items():
        cleaned_text = re.sub(pattern, replacement, cleaned_text)

    # ===== 6️⃣ 移除非打印字符（参考 postprocess_pred） =====
    np_pattern = re.compile(r'[\x00-\x1f]')
    cleaned_text = np_pattern.sub('', cleaned_text)

    # ===== 7️⃣ 规范化空格和括号 =====
    # 移除左括号后和右括号前的空格
    cleaned_text = re.sub(r'\(\s+', '(', cleaned_text)  # (  -> (
    cleaned_text = re.sub(r'\s+\)', ')', cleaned_text)  #  ) -> )
    cleaned_text = re.sub(r'\[\s+', '[', cleaned_text)  # [  -> [
    cleaned_text = re.sub(r'\s+\]', ']', cleaned_text)  #  ] -> ]
    cleaned_text = re.sub(r'\{\s+', '{', cleaned_text)  # {  -> {
    cleaned_text = re.sub(r'\s+\}', '}', cleaned_text)  #  } -> }
    # 移除逗号前后空格
    cleaned_text = re.sub(r'\s*,\s*', ',', cleaned_text)
    # 规范化多余空格
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # ===== 8️⃣ 错误处理：检查是否为空或无效 =====
    if not cleaned_text:
        return ""

    return cleaned_text

def Self_Certainty_Selection(dataset, config: Config, model, tokenizer, device,
                             N=4, save_results=False,
                             key_api_client: "OpenAI" = None,
                             key_api_model: str = "qwen3-max-preview",
                             key_api_temp: float = 0.2,
                             lambda_weight=0.5):
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

        # 提取真值答案（GSM8K样式 #### <ans>）
        match = re.search(r'####\s*(.+)', data['answer'])
        if match:
            true_answer = match.group(1).strip()
        else:
            true_answer = data['answer'].strip()

        question = data['question']

        # 构建模型提示
        # prompt = tokenizer.apply_chat_template(
        #     [{"role": "system", "content": config.system_prompt},
        #      {"role": "user", "content": f"Question: {question}\n\n\nAnswer:"}],
        #     tokenize=False, add_generation_prompt=True
        # )
        # prompt = tokenizer.apply_chat_template(
        #     [{"role": "user", "content": f"Question: {question}\nLet's think step by step and output the final answer after '####'.\n"}],
        #     tokenize=False, add_generation_prompt=True
        # )
        prompt = f"Question: {question}\nLet's think step by step and output the final answer after '####'.\n"

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
                # 使用新的基于熵的置信度计算方法，融入 lambda_weight
                step_confidence = calculate_entropy_based_confidence(
                    candidate["prompt_ids"],
                    candidate["tokens"],
                    candidate["logits"],
                    tokenizer,
                    lambda_weight=lambda_weight
                )
                step_confidence_scores.append(step_confidence)
                valid_candidates.append(candidate)

                response_text_dbg = tokenizer.decode(candidate["tokens"], skip_special_tokens=True)
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
        response_text = tokenizer.decode(best_candidate["tokens"], skip_special_tokens=True)

        # 提取模型答案用于准确率统计
        model_answer = extract_model_answer(response_text)
 
        # 更新统计
        n_samples += 1
        clean_key_step_text = clean_latex_format(response_text)
        if true_answer in clean_key_step_text[-10:] or is_correct_answer(model_answer, true_answer):
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

        # 仅使用 OpenAI 兼容接口抽取关键步骤
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

        # 保存为三字段格式
        table.append({
            "question": question,
            "answer": cleaned_text,
            "gpt_response": key_step_text,
            "max_confidence": step_confidence_scores[best_index],
            "correct": true_answer in clean_key_step_text[-10:] or is_correct_answer(model_answer, true_answer)
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
            print(f"关键步骤(gpt_response): {key_step_text}")
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
        output_file = f"./TTT_data/Best_of_{N}_Transformers_Entropy_Based_avglogp_Confidence_deepseek_7b_key_{config.lambda_weight}_temperature{config.temperature}.json"
        with open(output_file, mode="w", encoding="utf-8") as file:
            json.dump(table, file, indent=4, ensure_ascii=False)
        print(f"Results saved to {output_file}")


# ============================================================================
#                                Main
# ============================================================================

if __name__ == "__main__":
    # 创建配置对象
    config = Config()

    # 命令行参数
    parser = argparse.ArgumentParser(description="基于步骤置信度的评估方法（Transformers + GPT关键步骤提取）")

    # 生成相关
    parser.add_argument("--n_repetitive_sampling", default=4, type=int, help="为每个问题生成的解决方案数量")
    parser.add_argument("--temperature", default=0.1, type=float, help="生成时的温度参数，控制随机性")
    parser.add_argument("--top_p", default=1.0, type=float, help="Top-p采样参数")
    parser.add_argument("--model_path", default="meta-llama/Llama-3.2-1B-Instruct", help="基础模型路径")
    parser.add_argument("--save_to_json", default=True, action="store_true", help="是否将结果保存到JSON文件")
    parser.add_argument("--dataset_repo_name", default="openai/gsm8k", help="数据集仓库名称")
    parser.add_argument("--max_tokens", default=512, type=int, help="最大生成标记数")
    parser.add_argument("--subset_size", default=None, type=int, help="测试子集大小（用于快速测试）")
    parser.add_argument("--lambda_weight", default=0.5, type=float, help="步骤置信度和答案置信度的加权权重 (lambda for steps)")

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
        tokenizer.add_special_tokens({"pad_token": " [PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    model.eval()
    print("语言模型加载完成!")

    # 加载数据集
    print("正在加载数据集...")
    dataset = load_dataset(args.dataset_repo_name, 'main', split="test")

    # 测试子集
    if args.subset_size:
        dataset = dataset.select(range(min(args.subset_size, len(dataset))))

    print(f"数据集加载完成，共{len(dataset)}个样本")

    # 执行评估 + 保存为三字段
    print("开始执行基于步骤的置信度评估...")
    Self_Certainty_Selection(
        dataset,
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=device,
        N=config.n,
        save_results=args.save_to_json,
        key_api_client=key_client,
        key_api_model=args.key_step_api_model,
        key_api_temp=args.key_step_temperature,
        lambda_weight=args.lambda_weight
    )

    # 打印所有参数及其值
    print("########################################################################################")
    print("实验配置参数:")
    print("########################################################################################")
    for arg, value in vars(args).items():
        print(f"{arg:>25} ===> {value}")
    print("########################################################################################")