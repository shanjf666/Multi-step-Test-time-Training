#!/bin/bash

# ================= 配置区域 (请根据需要修改) =================

# 1. 模型列表 (HuggingFace Hub ID 或 本地绝对路径)
# 脚本会自动遍历这里列出的所有模型
MODELS=(
    "Qwen/Qwen2.5-Math-1.5B"
    # "Qwen/Qwen2.5-Math-1.5B-Instruct"
)

# 2. 数据集列表
# 脚本会自动遍历这里列出的所有数据集
DATASETS=(
    # "gsm8k"
    # "HuggingFaceH4/aime_2024"
    # "AI-MO/aimo-validation-amc"
    "HuggingFaceH4/MATH-500"
)

# 3. 数据集 Split (默认通常为 test)
DATASET_SPLIT="test"

# 4. 采样参数 (建议 N=64 以发挥 DoubleStab 优势)
N_SAMPLES=64

# 5. 硬件配置 (vLLM Tensor Parallel Size)
GPU_COUNT=1

# 6. 输出目录
OUTPUT_DIR="benchmark_results_multi"

# ==========================================================

# 错误处理
set -e

# 获取当前时间戳 (用于本次批次)
BATCH_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$OUTPUT_DIR"

echo "################################################################"
echo "#          DoubleStab Multi-Model Benchmark Pipeline           #"
echo "################################################################"
echo "Models:     ${#MODELS[@]} models scheduled"
echo "Datasets:   ${#DATASETS[@]} datasets scheduled"
echo "Samples:    $N_SAMPLES"
echo "GPUs:       $GPU_COUNT"
echo "Log Dir:    $OUTPUT_DIR"
echo "################################################################"
echo ""

# 循环遍历模型
for model_path in "${MODELS[@]}"; do
    # 循环遍历数据集
    for dataset_name in "${DATASETS[@]}"; do
        
        echo "=================================================================="
        echo ">>> Processing Task: [$model_path] on [$dataset_name]"
        echo "=================================================================="

        # --- 0. 生成文件名 ---
        # 将模型名和数据集名中的 '/' 替换为 '_' 以免破坏文件路径
        SAFE_MODEL_NAME=$(echo "$model_path" | tr '/' '_')
        SAFE_DS_NAME=$(echo "$dataset_name" | tr '/' '_')
        
        TASK_ID="${SAFE_MODEL_NAME}_${SAFE_DS_NAME}"
        OUTPUT_FILE="${OUTPUT_DIR}/${TASK_ID}_${BATCH_TIMESTAMP}.jsonl"
        LOG_FILE="${OUTPUT_DIR}/${TASK_ID}_${BATCH_TIMESTAMP}.log"

        # 记录开始时间
        START_TIME=$SECONDS

        # --- 1. 生成回复并计算指标 ---
        echo ">>> [Step 1/2] Generation (vLLM)..."
        echo "    Output: $OUTPUT_FILE"
        
        # 使用 tee 同时输出到屏幕和日志文件
        python run_generation.py \
            --model "$model_path" \
            --dataset "$dataset_name" \
            --split "$DATASET_SPLIT" \
            --output_file "$OUTPUT_FILE" \
            --n_samples "$N_SAMPLES" \
            --gpu_count "$GPU_COUNT" 2>&1 | tee -a "$LOG_FILE"

        GEN_DURATION=$((SECONDS - START_TIME))
        echo ">>> Generation finished in ${GEN_DURATION}s." | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"

        # --- 2. 评估策略 ---
        echo ">>> [Step 2/2] Evaluating Strategies..." | tee -a "$LOG_FILE"

        python run_evaluation.py \
            --input_file "$OUTPUT_FILE" 2>&1 | tee -a "$LOG_FILE"

        echo "" | tee -a "$LOG_FILE"
        echo ">>> Task Completed: $TASK_ID" | tee -a "$LOG_FILE"
        echo "------------------------------------------------------------------"
        echo ""

    done
done

echo "################################################################"
echo "#                  All Benchmarks Completed                    #"
echo "################################################################"
echo "Check results in: $OUTPUT_DIR"