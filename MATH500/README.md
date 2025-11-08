```markdown|CODE_EDIT_BLOCK|c:\Users\13726\Desktop\科研\Multi-step Test-time Training\Multi-step-Test-time-Training\MATH500\README.md
# MATH500 快速推理工具

本项目提供了一个优化的推理流程，通过`infer.py`脚本可以快速地对MATH-500数据集进行测试时训练推理。

## 简介

`infer.py`是一个集成脚本，它将候选答案生成和评估选择两个步骤合并在一起执行，相比之前的方法更加高效。该脚本会自动调用`generate.py`生成候选答案，然后使用`evaluate.py`对候选答案进行评估并选择最优答案。

## 使用方法

### 基本用法

```bash
python infer.py --model_path meta-llama/Llama-3.2-1B-Instruct
```

### 参数说明

- `--model_path` (必需): HuggingFace模型路径
- `--num_return_sequences` (可选): 每个问题生成的候选答案数量，默认为4
- `--temperature` (可选): 采样温度，控制生成的随机性，默认为0.1
- `--max_tokens` (可选): 生成的最大token数，默认为1024
- `--alpha` (可选): 在evaluate.py中使用的self-certainty加权系数，默认为0.4

### 示例命令

使用Qwen2.5-Math-7B模型进行推理：
```bash
python infer.py --model_path Qwen/Qwen2.5-Math-7B-Instruct --num_return_sequences 8 --temperature 0.7 --max_tokens 2048 --alpha 0.5
```

使用Llama-3.2-1B模型进行推理：
```bash
python infer.py --model_path meta-llama/Llama-3.2-1B-Instruct
```

使用不同的参数组合进行推理：
```bash
python infer.py --model_path Qwen/Qwen2.5-Math-1.5B-Instruct --num_return_sequences 4 --temperature 0.3 --alpha 0.4
```

## 工作流程详解

完整的推理流程分为两个主要阶段：

### 第一阶段：候选答案生成 (generate.py)

1. 加载指定的预训练模型和对应的tokenizer
2. 从HuggingFace加载MATH-500测试数据集
3. 为每个数学问题构建chat-style提示词模板：
   ```
   Q: {问题内容}
   Let's think step by step and output the final answer within \boxed{}
   A:
   ```
4. 使用vLLM引擎并行生成多个候选答案（数量由`--num_return_sequences`参数控制）
5. 将所有生成的候选答案保存为JSONL格式文件，命名为：
   `results/math500_candidates_{模型名}_{时间戳}.jsonl`

### 第二阶段：答案评估与选择 (evaluate.py)

1. 读取第一阶段生成的候选答案文件
2. 加载与生成阶段相同的模型和tokenizer用于评估
3. 对每个问题的多个候选答案分别计算置信度分数：
   - 使用self-certainty方法计算每个token的置信度
   - 使用negative-entropy方法计算每个token的置信度
   - 结合token级置信度和步骤级置信度得到综合评分
   - 利用`--alpha`参数平衡两种置信度的权重
4. 为每个问题选择置信度分数最高的候选答案作为最终答案
5. 将最终结果保存为JSONL格式文件，命名为：
   `results/math500_best_{模型名}_{时间戳}.jsonl`

### 整体流程自动化

`infer.py`脚本会按顺序自动执行上述两个阶段，并在完成后显示类似以下的输出：

```
[1/2] Generating candidates -> results/math500_candidates_Qwen2.5-Math-7B-Instruct_20250406_120000.jsonl
[2/2] Evaluating and selecting best answers -> results/math500_best_Qwen2.5-Math-7B-Instruct_20250406_120000.jsonl
✅ Done! Final results saved to: results/math500_best_Qwen2.5-Math-7B-Instruct_20250406_120000.jsonl
```

## 输出格式

生成的结果文件为JSONL格式，每一行包含一个问题及其最优答案：

```json
{"question": "问题文本", "answer": "模型生成的最佳答案"}
```

## 性能优势

相比于原始的多步骤方法，`infer.py`具有以下优势：

1. **一体化流程**: 将生成和评估步骤整合在一个脚本中，简化操作流程
2. **更高的效率**: 优化了数据传递过程，减少了中间文件的读写开销
3. **更好的易用性**: 只需要一条命令即可完成整个推理过程

## 内存优化版本

为了应对大规模模型和大批量数据处理的需求，我们开发了基于`infer.py`的内存优化版本。该版本采用以下优化策略：

### 优化特性

1. **流式处理**: 采用流式数据处理方式，避免将全部数据加载到内存中
2. **即时清理**: 在处理过程中及时释放不再需要的中间变量和缓存
3. **分批处理**: 将大数据集分成小批次处理，降低峰值内存使用量
4. **vLLM集成**: 利用vLLM的高效推理能力减少显存占用

### 完整功能版本 (main.py)

对于需要更全面分析和多种评估方法的场景，我们也提供了基于`main.py`的完整功能版本。该版本虽然在内存使用和运行速度方面不如`infer.py`优化版本，但具备以下特点：

### 支持的评估方法

1. **Baseline**: 基础方法，每个问题只生成一个答案
2. **Entropy**: 基于生成过程中token级别熵值评估答案质量
3. **CoE-C**: 基于链式嵌入收敛性评估答案质量
4. **Self-Certainty**: 结合token级别和步骤级别的置信度评估
5. **Self-Eval**: 使用模型自身评估每个推理步骤的正确性
6. **Self-Consistency**: 基于多数投票选择最一致的答案

### 使用方法

```bash
python main.py --method self-certainty --model_path Qwen/Qwen2.5-Math-7B-Instruct --lambda_weight 0.5
python main.py --method coe-c --model_path Qwen/Qwen2.5-Math-7B-Instruct --temperature 0.7 --n_repetitive_sampling 8
python main.py --method entropy --model_path Qwen/Qwen2.5-Math-1.5B-Instruct --temperature 0.1
```

### 特点

- 支持更多评估指标和分析维度
- 提供详细的调试信息和中间结果
- 支持关键步骤提取和分析
- 适用于深入研究和算法比较

## 注意事项

1. 确保已安装所有必要的依赖项（可通过上级目录的`requirements.txt`安装）
2. 确保系统有足够的GPU内存来运行指定的模型
3. 生成的结果将保存在项目目录下的`results/`文件夹中
```