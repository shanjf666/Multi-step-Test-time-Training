```markdown
# Multi-step Test-time Training with Data Selection

本项目实现了多种测试时训练(Test-time Training)方法，通过生成多个候选答案并选择最优解，为后续强化学习提供高质量训练数据。

## 项目概述

在测试阶段，模型为每个问题生成多个候选答案，通过不同评估方法计算置信度分数，选择置信度最高的答案作为最终输出。这些高质量的问答对可用于后续的强化学习训练。

## 三种评估方法

### 1. CoE-C 方法 (Chain-of-Embedding Convergence)
文件: [CoE-C-deepseek.py]

基于链式嵌入收敛性评估答案质量：
- 分析生成过程中隐藏状态的轨迹变化
- 计算相邻层之间的幅度和角度变化
- 使用CoE-C得分作为置信度指标

### 2. 自置信度方法 (Self-Certainty)
文件: [self-certainty-deepseek.py]

结合token级别和步骤级别的置信度：
- 使用概率分布的熵和对数似然计算置信度
- 平衡token置信度和步骤概率

### 3. 自我评估方法 (Self-Evaluation)
文件: [self-eval-deepseek.py]

使用模型自身评估每个推理步骤：
- 让模型判断每个步骤的正确性(A/B选项)
- 基于模型自评分数计算整体置信度

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### CoE-C 方法
```bash
python CoE-C-deepseek.py --model_path /path/to/model --lambda_weight 0.5
```

### 自置信度方法
```bash
python self-certainty-deepseek.py --model_path /path/to/model --lambda_weight 0.5
```

### 自我评估方法
```bash
python self-eval-deepseek.py --model_path /path/to/model --lambda_weight 0.5
```

## 主要参数

- `--model_path`: 预训练模型路径
- `--subset_size`: 处理样本数量(用于快速测试)
- `--lambda_weight`: 权重参数，平衡不同置信度指标(0-1)
- `--n_repetitive_sampling`: 每个问题生成的候选答案数量
- `--temperature`: 生成时的温度参数(控制随机性)
- `--max_tokens`: 最大生成token数

## 工作流程

1. **生成候选答案**: 为每个问题生成N个候选答案
2. **计算置信度**: 使用不同方法计算每个候选答案的置信度分数
3. **选择最优解**: 选择置信度最高的答案作为最终输出
4. **保存结果**: 将高质量问答对保存为JSON格式，用于RL训练

## 输出格式

生成的JSON文件包含以下信息：
```json
{
  "ID": 1,
  "model_input": "问题文本",
  "output": ["模型生成的完整答案"],
  "model_answer": "提取的答案",
  "true_answer": "真实答案",
  "confidence": 0.85,  // 分数
  "is_correct": true   // 答案正确性
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
