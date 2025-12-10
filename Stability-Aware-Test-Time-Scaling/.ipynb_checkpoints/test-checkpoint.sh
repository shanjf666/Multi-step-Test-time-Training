#!/bin/bash

# ================= 快速测试配置 =================
# 使用一个小模型，或者你已经下载好的模型路径
MODEL_PATH="Qwen/Qwen2.5-Math-1.5B"

# 这是一个微型测试：只跑 5 个样本，每个生成 4 条
# 这样几分钟内就能跑完，验证代码逻辑是否通畅
DATASET_NAME="gsm8k"
DATASET_SPLIT="test"
MAX_SAMPLES=5
N_SAMPLES=4
GPU_COUNT=1

OUTPUT_DIR="test_results"
# ===============================================

set -e # 遇到错误立即退出

echo ">>> [Init] Checking environment..."
mkdir -p "$OUTPUT_DIR"

# 检查 Python 脚本是否存在
for file in "math_utils.py" "run_generation.py" "run_evaluation.py"; do
    if [ ! -f "$file" ]; then
        echo "Error: $file not found!"
        exit 1
    fi
done

TIMESTAMP=$(date +"%H%M%S")
TEST_OUTPUT="${OUTPUT_DIR}/test_run_${TIMESTAMP}.jsonl"

echo ""
echo "#####################################################"
echo "#           DoubleStab Pipeline: DRY RUN            #"
echo "#####################################################"
echo "Model:       $MODEL_PATH"
echo "Max Samples: $MAX_SAMPLES (Tiny subset)"
echo "N Samples:   $N_SAMPLES (Quick rollout)"
echo "Output:      $TEST_OUTPUT"
echo "#####################################################"
echo ""

# --- Step 1: 生成 ---
echo ">>> [Step 1] Running Generation (Mini Batch)..."
python run_generation.py \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_NAME" \
    --split "$DATASET_SPLIT" \
    --output_file "$TEST_OUTPUT" \
    --n_samples "$N_SAMPLES" \
    --max_samples "$MAX_SAMPLES" \
    --gpu_count "$GPU_COUNT"

echo ">>> Generation done."
echo ""

# --- Step 2: 评估 ---
echo ">>> [Step 2] Running Evaluation..."
python run_evaluation.py \
    --input_file "$TEST_OUTPUT"

echo ""
echo "#####################################################"
echo "#                 Test Completed!                   #"
echo "#####################################################"
echo "If you see the leaderboard above, the code is working."
echo "You can now run 'run_benchmark.sh' for the full experiment."