# LLM Reasoning & Test-Time Adaptation

本项目是一个关于大语言模型（LLM）推理能力研究的综合代码库，专注于 **思维链（Chain-of-Thought, CoT）** 的评估、**测试时计算（Test-Time Compute）** 扩展以及**无奖励模型（Reward-Free）** 的数据筛选策略。

项目包含三个核心模块，分别侧重于深度指标分析、强化学习数据生成以及 SASC 算法实现。

## 📂 项目结构 (Directory Structure)

### 1\. [CoT-Analyze-Toolkit](https://www.google.com/search?q=./CoT-Analyze)

**👉 定位：推理能力评估与深度指标分析工具箱**

这是一个基础分析套件，用于对 LLM 的推理路径进行细粒度的“体检”。

  * **核心功能**：利用 `vLLM` 进行多路采样，计算 Token 级和 Step 级的**熵 (Entropy)**、**确信度**和**困惑度**。
  * **应用场景**：分析模型在哪些步骤容易产生幻觉，或者基于一致性分数（SC Score）将数据划分为“简单/困难”桶，用于课程学习。

### 2\. [Multi-Step-Test-Time-Scaling](https://www.google.com/search?q=./Multi-Step-Test-Time-Scaling)

**👉 定位：测试时训练（TTT）与 RL 数据生成**

该模块实现了多种测试时适应方法，旨在为后续的强化学习（RL）提供高质量的训练数据。

  * **核心方法**：实现了 **CoE-C** (Chain of Embeddings Convergence)、**Self-Certainty**、**Self-Evaluation** 等多种置信度评估算法。
  * **应用场景**：生成并筛选高质量的问答对（Question-Answer Pairs），用于 PPO/GRPO 训练。

### 3\. [SASC-Test-Time-Scaling](https://www.google.com/search?q=./SASC-Test-Time-Scaling)

**👉 定位：SASC 算法与 Combo 策略实现**

这是本项目最核心的算法实现部分，专注于 **Stability-Aware Self-Consistency (SASC)** 及其衍生策略。

  * **核心创新**：无需 Ground Truth，利用**熵稳定性**和**双重加权**机制筛选最佳答案。
  * **关键策略**：包含 **Combo 策略**（FilterTopK + Weighted），在 GSM8K/MATH 等基准测试上显著优于传统的 Majority Voting。

-----

## 🚀 快速导航 (Quick Navigation)

| 模块名称 | 关键词 | 适合场景 |
| :--- | :--- | :--- |
| **CoT-Analyze** | `Entropy Analysis`, `Data Bucketing`, `Difficulty Filter` | 需要分析模型行为，或清洗数据用于微调时。 |
| **Multi-Step-TTT** | `RL Data Generation`, `CoE-C`, `Self-Eval` | 准备强化学习数据集，或对比不同 TTT 方法时。 |
| **SASC-Scaling** | `SASC`, `Combo Strategy`, `Reward-Free TTA` | **研究核心**。在推理阶段追求最高准确率，验证 SASC/Combo 效果时。 |

## 🛠️ 核心依赖

整个项目主要基于以下技术栈，请确保环境满足要求：

  * **Inference**: `vLLM` (用于高并发推理)
  * **Core**: `PyTorch`, `Transformers`
  * **Data**: `Datasets`, `NumPy`

## 📖 使用说明

每个子文件夹下均包含独立的 `README.md` 和运行脚本。请点击上方链接进入相应目录查看详细文档。

```bash
# 示例：运行 SASC 策略评估
cd SASC-Test-Time-Scaling
./run_benchmark.sh
```

-----

*License: MIT*