import re

def clean_latex_format(text) -> str:
    """
    清理 LaTeX 格式，标准化数学符号。
    兼容处理 int/float 类型的输入。
    """
    if text is None:
        return ""
    
    # [关键修改] 强制转换为字符串，防止 AIMO 等数据集传入 int/float 导致崩溃
    text = str(text)
    
    text = text.strip()
    # 基础清理
    text = text.replace("$", "").replace("\\$", "$")
    text = text.replace(r'\(', '(').replace(r'\)', ')')
    text = text.replace(r'\[', '[').replace(r'\]', ']')
    
    # 移除无用的 LaTeX 命令
    useless_cmds = [
        r'\\left', r'\\right', r'\\big', r'\\Big', r'\\bigg', r'\\Bigg',
        r'\\text\s*', r'\\mathrm', r'\\displaystyle', r'\\rm', r'\\it',
        r'\\bf', r'\\cal'
    ]
    for cmd in useless_cmds:
        text = re.sub(cmd, '', text)
        
    # 标准化空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_answer(text: str) -> str:
    """从回复中提取答案，支持 \\boxed{} 和常见格式"""
    if not text: return ""
    
    # 1. 尝试提取 \boxed{} (最优先)
    idx = text.rfind("\\boxed")
    if idx >= 0:
        content = ""
        open_braces = 0
        started = False
        for i in range(idx, len(text)):
            char = text[i]
            if char == "{":
                open_braces += 1
                started = True
                if open_braces == 1: continue
            elif char == "}":
                open_braces -= 1
                if open_braces == 0 and started:
                    return content.strip()
            
            if started and open_braces > 0:
                content += char
                
    # 2. 尝试提取 "The answer is" 等模式
    patterns = [
        r"The answer is\s*(.*)",
        r"Answer:\s*(.*)",
        r"####\s*(.*)" 
    ]
    for p in patterns:
        match = re.search(p, text, re.IGNORECASE)
        if match:
            clean = match.group(1).strip().strip('.')
            if len(clean) > 50:
                 last_part = clean.split()[-1]
                 return last_part
            return clean

    # 3. 兜底：返回最后一行
    lines = text.strip().split('\n')
    last_line = lines[-1].strip()
    
    if len(last_line) < 100: 
        if '=' in last_line:
            return last_line.split('=')[-1].strip().strip('.')
        return last_line.strip('.')
        
    return ""

def strip_string(string):
    """标准化答案字符串用于比较"""
    if string is None: return ""
    string = str(string)
    
    # 1. 移除 LaTeX 包装
    string = string.replace("\\boxed", "").replace("{", "").replace("}", "")
    string = string.replace("\\(", "").replace("\\)", "")
    
    # 2. 移除常见的数学符号干扰
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace(" ", "")
    
    # 3. 移除千分位逗号
    if string.replace(',', '').replace('.', '').isdigit():
        string = string.replace(',', '')
        
    # 4. 统一分数 (简单处理)
    if string == "0.5": string = "\\frac{1}{2}"
    
    # 5. 移除末尾句号
    string = string.rstrip('.')
    
    return string

def is_correct(pred: str, gt: str) -> bool:
    """
    判断预测是否正确，支持数值容错比较。
    解决 '18' != '18.0' 的问题。
    """
    # 1. 字符串归一化
    norm_pred = strip_string(clean_latex_format(pred))
    norm_gt = strip_string(clean_latex_format(gt))
    
    # 2. 尝试字符串精确匹配
    if norm_pred == norm_gt:
        return True
        
    # 3. [新增] 尝试数值匹配
    # 如果两者都能转换为浮点数，比较数值差异
    try:
        float_pred = float(norm_pred)
        float_gt = float(norm_gt)
        # 允许 1e-6 的误差
        return abs(float_pred - float_gt) < 1e-6
    except ValueError:
        pass
        
    return False

def parse_steps(text: str):
    """将文本切分为步骤"""
    steps = []
    step_pattern = re.compile(r'(?:##\s*)?Step\s*(?:\d+)\s*[:.]?\s*(.*?)(?=(?:##\s*)?Step\s*\d+\s*[:.]?|$)', re.DOTALL | re.IGNORECASE)
    matches = list(step_pattern.finditer(text))
    if matches:
        return [m.group(1).strip() for m in matches if m.group(1).strip()]

    parts = text.split('\n\n')
    steps = [p.strip() for p in parts if p.strip()]
    
    if len(steps) < 2:
        parts = text.split('\n')
        steps = [p.strip() for p in parts if len(p.strip()) > 10] 

    return steps if steps else [text]