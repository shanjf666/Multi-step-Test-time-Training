#!/bin/bash
# 智能多GPU任务调度脚本（支持真正并行运行，无OOM重试）
DATASET="gsm8k"
MODEL="Qwen7B"
LOG_DIR="./logs"
OUTPUT_DIR="./TTT_data"
mkdir -p $LOG_DIR $OUTPUT_DIR
METHODS=("baseline" "self-consistency" "entropy" "self-certainty" "self-eval" "coe-c")
SAMPLE_SIZES=(4)
LAMBDA_WEIGHTS=(0.1 0.3 0.5 0.7 0.9)
NUM_GPUS=8
CHECK_INTERVAL=10
echo "🚀 开始自动调度GSM8K实验..."
echo "GPU数: $NUM_GPUS, 每卡最大任务: 1"

# 清理旧锁文件
rm -f /tmp/gpu_lock_*

# ========== 获取一个空闲的 GPU（无锁文件） ==========
get_available_gpu() {
    for i in $(seq 0 $((NUM_GPUS-1))); do
        if [ ! -f /tmp/gpu_lock_$i ]; then
            echo "$i"
            return 0
        fi
    done
    echo ""
}

# ========== 执行函数（无重试） ==========
run_task() {
    local method="$1"
    local N="$2"
    local LAMBDA="$3"
    local log="$4"
    local gpu_id="$5"

    # 创建锁文件
    touch /tmp/gpu_lock_$gpu_id

    echo "在 GPU $gpu_id 上启动任务..."
    # 根据方法构造命令
    if [[ "$method" == "base" ]]; then
        cmd="CUDA_VISIBLE_DEVICES=$gpu_id python base.py --n_repetitive_sampling $N --max_tokens 1024"
    elif [ -z "$LAMBDA" ]; then
        cmd="CUDA_VISIBLE_DEVICES=$gpu_id python main.py --method $method --n_repetitive_sampling $N --max_tokens 1024"
    else
        cmd="CUDA_VISIBLE_DEVICES=$gpu_id python main.py --method $method --n_repetitive_sampling $N --lambda_weight $LAMBDA --max_tokens 1024"
    fi
    echo "[执行任务] $cmd"
    eval "$cmd" >> "$log" 2>&1
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "⚠️ 程序异常退出 (code=$exit_code)，任务失败 (GPU $gpu_id)"
        rm /tmp/gpu_lock_$gpu_id
        return 1
    else
        echo "✅ 任务成功完成 (GPU $gpu_id)"
        rm /tmp/gpu_lock_$gpu_id
        return 0
    fi
}

# ========== 任务提交函数（并行执行） ==========
submit_task() {
    local method=$1
    local N=$2
    local LAMBDA=$3
    local log_file gpu_id
    # 等待直到找到一个可用 GPU
    gpu_id=""
    while [ -z "$gpu_id" ]; do
        gpu_id=$(get_available_gpu)
        if [ -z "$gpu_id" ]; then
            echo "所有 GPU 忙碌，等待 $CHECK_INTERVAL 秒..."
            sleep $CHECK_INTERVAL
        fi
    done
    # 构造日志路径
    if [[ "$method" == "base" ]]; then
        log_file=${LOG_DIR}/base_N${N}_${MODEL}_${DATASET}.log
        echo "[GPU $gpu_id] 启动 base.py (N=${N})"
    elif [ -z "$LAMBDA" ]; then
        log_file=${LOG_DIR}/${method}_N${N}_${MODEL}_${DATASET}.log
        echo "[GPU $gpu_id] 启动 ${method} (N=${N})"
    else
        log_file=${LOG_DIR}/${method}_N${N}_lambda${LAMBDA}_${MODEL}_${DATASET}.log
        echo "[GPU $gpu_id] 启动 ${method} (N=${N}, λ=${LAMBDA})"
    fi
    # 并行后台运行任务
    (
        run_task "$method" "$N" "$LAMBDA" "$log_file" "$gpu_id"
    ) &
    sleep 1  # 短暂等待以避免锁竞争
}

# ========== 提交任务 ==========
submit_task "baseline" 1 ""
for N in "${SAMPLE_SIZES[@]}"; do
    submit_task "self-consistency" $N ""
    submit_task "base" $N ""
done
for method in "entropy" "self-certainty" "self-eval" "coe-c"; do
    for N in "${SAMPLE_SIZES[@]}"; do
        for LAMBDA in "${LAMBDA_WEIGHTS[@]}"; do
            submit_task "$method" $N $LAMBDA
        done
    done
done
# 等待所有任务结束
wait
echo "✅ 所有任务完成！"
echo "日志保存在：$LOG_DIR"
echo "输出保存在：$OUTPUT_DIR"