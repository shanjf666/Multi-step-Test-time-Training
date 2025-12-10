# SASC: Stability-Aware Self-Consistency Test Time Scaling

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()
[![vLLM](https://img.shields.io/badge/Inference-vLLM-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

## 📖 简介 (Introduction)

本项目实现了一个**无需训练奖励模型 (Reward-Free)** 的大模型数学推理评估框架，旨在探索 **Test-Time Adaptation (TTA)** 的新方法。

核心算法 **SASC (Stability-Aware Self-Consistency)** 利用模型推理过程中的**内在不确定性 (Intrinsic Uncertainty)**——具体为 Token 级别的熵 (Entropy) 和 步骤级别的稳定性 (Step Stability)——来衡量 CoT 推理路径的质量，从而在无需 Ground Truth 的情况下筛选出最佳答案。

本框架包含完整的流水线：
1.  **高效生成**：基于 `vLLM` 的高并发推理，支持实时计算 Logprobs 和 Entropy。
2.  **指标提取**：自动计算 `Avg Entropy`、`Std Entropy` 以及创新性的 `Step-level Stability`。
3.  **鲁棒评估**：内置强大的数学答案归一化工具，支持多种加权投票策略。

## ✨ 核心特性 (Key Features)

* **🚀 高性能推理**: 集成 `vLLM`，支持 Tensor Parallel 和 Chunk Processing，在单卡/多卡上高效处理 GSM8K/MATH/AIMO 等大规模数据集。
* **📊 多维不确定性指标**:
    * **Token Entropy**: 衡量模型在每个 Token 上的犹豫程度。
    * **Step-level Stability**: 衡量模型在不同推理步骤间确信度的波动情况（SASC 核心假设）。
* **🧮 强大的数学解析**: 解决了 `142.0` vs `142`、`3/4` vs `0.75`、`\frac{1}{2}` vs `0.5` 等棘手的格式匹配问题，显著提升 Pass@1 和 SC 的评估准确性。
* **⚖️ 多种选择策略**:
    * `Baseline: Consistency (SC)`: 标准众数投票。
    * `Weighted: Z-Std-Entropy`: 基于整体熵的加权。
    * `Hybrid: Double Stability`: **(SASC)** 结合整体熵与步骤稳定性的双重加权。
    * `Combo`: 先过滤掉高熵路径，再进行加权。


## 🚀 快速开始 (Quick Start)

### 方式一：一键运行 Benchmark (推荐)

使用 `run_benchmark.sh` 脚本可以自动遍历多个模型和数据集。

1.  编辑 `run_benchmark.sh` 配置你的模型路径和数据集：
    ```bash
    MODELS=("Qwen/Qwen2.5-Math-1.5B")
    DATASETS=("AI-MO/aimo-validation-amc" "gsm8k")
    ```
2.  运行脚本：
    ```bash
    chmod +x run_benchmark.sh
    ./run_benchmark.sh
    ```
    结果将保存在 `benchmark_results_multi/` 目录下。

### 方式二：分步运行

**Step 1: 生成回复与计算指标**

```bash
python run_generation.py \
    --model Qwen/Qwen2.5-Math-1.5B \
    --dataset gsm8k \
    --split test \
    --output_file results/gsm8k_output.jsonl \
    --n_samples 64 \
    --gpu_count 1
```

**Step 2: 评估策略效果**

```bash
python run_evaluation.py --input_file results/gsm8k_output.jsonl
```

## 📂 文件结构 (File Structure)

```text
.
├── run_generation.py    # [生成器] 调用 vLLM 推理，计算 Logprobs、熵和置信度
├── run_evaluation.py    # [评估器] 解析答案，执行加权投票策略，生成报表
├── math_utils.py        # [工具库] 包含 LaTeX 清洗、答案提取、数值归一化逻辑
├── run_benchmark.sh     # [调度器] 批处理脚本，自动化运行整个 Benchmark
└── README.md            # 项目文档
```

## 🧠 方法论 (Methodology)

本项目对比了以下几种推理路径选择策略：

| 策略名称 | 描述 | 公式/逻辑 |
| :--- | :--- | :--- |
| **Pass@1** | 贪婪解码或单次采样 | $P(\text{greedy})$ |
| **Consistency (SC)** | Self-Consistency (众数投票) | $\text{argmax} \sum \mathbb{I}(y_i = c)$ |
| **Z-Std-Entropy** | 基于熵的标准差加权。熵越低，权重越大。 | $w_i = \exp(-Z_{\text{entropy}})$ |
| **Double Stability** | **(SASC)** 结合整体熵与步骤间熵的稳定性。 | $w_i = \exp(-Z_{\text{entropy}} - Z_{\text{step\_std}})$ |
| **Combo** | 先过滤掉熵最高的 K% (FilterTopK)，再进行加权。 | $\text{Filter}(Z_{\text{entropy}} > \tau) \rightarrow \text{Weighted}$ |

## 📊 结果示例 (Sample Output)

运行 `run_evaluation.py` 后，控制台将输出如下报告：

```text
================================================================================
                          GENERALIZATION BENCHMARK REPORT                       
================================================================================
Strategy                            | Accuracy   | vs SC     
--------------------------------------------------------------------------------
Combo: FilterTopK + W-StdTopK       | 69.10%     | +2.70% 👑
Hybrid: Double Stability            | 68.50%     | +2.10% 
Weighted: Z-Std-Entropy             | 67.80%     | +1.40% 
Baseline: Consistency (SC)          | 66.40%     | +0.00% 
Baseline: Pass@1                    | 52.10%     | -14.30%
================================================================================
```


## 📄 License

This project is licensed under the MIT License.

```
```