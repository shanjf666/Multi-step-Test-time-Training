# Multi-step Test-time Training with Data Selection

本项目实现了多种测试时训练(Test-time Training)方法，通过生成多个候选答案并选择最优解，为后续强化学习提供高质量训练数据。

## 项目概述

在测试阶段，模型为每个问题生成多个候选答案，通过不同评估方法计算置信度分数，选择置信度最高的答案作为最终输出。这些高质量的问答对可用于后续的强化学习训练。

## 支持的方法

### 1. Baseline 方法
文件: `methods/baseline.py`

基础方法，每个问题只生成一个答案，不进行置信度评估。

### 2. CoE-C 方法 
文件: `methods/coe_c.py`

基于链式嵌入收敛性评估答案质量：
- 分析生成过程中隐藏状态的轨迹变化
- 计算相邻层之间的幅度和角度变化
- 使用CoE-C得分作为置信度指标

### 3. Self-Certainty方法
文件: `methods/self_certainty.py`

结合token级别和步骤级别的置信度：
- 使用概率分布的熵和对数似然计算置信度
- 平衡token置信度和步骤概率

### 4. Self-Evaluation方法
文件: `methods/self_eval.py`

使用模型自身评估每个推理步骤：
- 让模型判断每个步骤的正确性
- 基于模型自评分数计算整体置信度

### 5. Self-Consistency方法
文件: `methods/self_consistency.py`

基于多数投票的评估方法：
- 生成多个候选答案
- 选择出现频率最高的答案作为最终输出

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python main.py --method [method_name] --model_path /path/to/model
```

### 参数说明

- `--method`: 选择评估方法 (baseline, self-certainty, self-eval, coe-c, self-consistency)
- `--model_path`: 预训练模型路径
- `--n_repetitive_sampling`: 每个问题生成的候选答案数量 (默认: 4)
- `--temperature`: 生成时的温度参数，控制随机性 (默认: 0.1)
- `--top_p`: Top-p采样参数 (默认: 1.0)
- `--max_tokens`: 最大生成token数 (默认: 512)
- `--subset_size`: 处理样本数量(用于快速测试)
- `--lambda_weight`: 权重参数，平衡不同置信度指标(0-1) (默认: 0.5)
- `--dataset_repo_name`: 数据集仓库名称 (默认: "openai/gsm8k")

### 关键步骤提取参数

- `--key_step_api_base`: OpenAI兼容API的基础URL (默认: "https://api.siliconflow.cn/v1")
- `--key_step_api_model`: 用于关键步骤提取的模型 (默认: "Qwen/Qwen2.5-7B-Instruct")
- `--key_step_temperature`: 关键步骤提取时的温度参数 (默认: 0.2)

### 示例命令

#### Baseline 方法
```bash
python main.py --method baseline --model_path meta-llama/Llama-3.2-1B-Instruct
```

#### CoE-C 方法
```bash
python main.py --method coe-c --model_path meta-llama/Llama-3.2-1B-Instruct --lambda_weight 0.5
```

#### Self-Certainty方法
```bash
python main.py --method self-certainty --model_path meta-llama/Llama-3.2-1B-Instruct --lambda_weight 0.5
```

#### Self-Evaluation方法
```bash
python main.py --method self-eval --model_path meta-llama/Llama-3.2-1B-Instruct --lambda_weight 0.5
```

#### Self-Consistency方法
```bash
python main.py --method self-consistency --model_path meta-llama/Llama-3.2-1B-Instruct
```

## 工作流程

1. **生成候选答案**: 为每个问题生成N个候选答案
2. **计算置信度**: 使用不同方法计算每个候选答案的置信度分数
3. **选择最优解**: 选择置信度最高的答案作为最终输出
4. **保存结果**: 将高质量问答对保存为JSON格式，用于RL训练

## 输出格式

生成的JSON文件包含以下信息：
```json
{
  "question": "问题文本",
  "answer": "模型生成的完整答案",
  "gpt_response": "提取的关键步骤"
}
```

## RL训练应用

保存的高质量问答对可直接用于强化学习训练：
- **置信度分数**: 作为奖励信号的参考
- **多样本生成**: 提供丰富的训练数据
- **质量保证**: 筛选出高质量的推理路径
- **错误分析**: 对比不同方法的选择结果

## 输出目录

结果保存在 `./TTT_data/` 目录中

## 项目结构

```
.
├── core/                 # 核心配置
├── methods/              # 各种TTT方法实现
├── utils/                # 工具函数
├── main.py              # 主入口点
└── README.md            # 项目说明
```