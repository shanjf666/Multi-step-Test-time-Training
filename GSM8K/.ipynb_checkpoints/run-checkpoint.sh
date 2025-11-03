#!/bin/bash
# 智能多GPU任务调度脚本
# 自动分配空闲GPU运行任务，不需手动指定设备号

DATASET="gsm8k"
MODEL="Qwen7B"
LOG_DIR="./logs"
OUTPUT_DIR="./TTT_data"
mkdir -p $LOG_DIR $OUTPUT_DIR

METHODS=("baseline" "self-consistency" "entropy" "self-certainty" "self-eval" "coe-c")
SAMPLE_SIZES=(1 2 4 8)
LAMBDA_WEIGHTS=(0.1 0.3 0.5 0.7 0.9)
NUM_GPUS=4                 # 总GPU数量
MAX_JOBS=4                 # 最大同时运行任务数（通常与NUM_GPUS相同）
CHECK_INTERVAL=10          # 检查空闲GPU的时间间隔(s)

echo "开始自动调度GSM8K实验..."

# 获取最空闲的GPU编号（利用nvidia-smi）
get_free_gpu() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
    | awk '{
        if (NR==1 || $1<min) {min=$1; id=NR-1}
     } END {print id}'
}

# 等待空闲GPU出现
wait_for_free_gpu() {
    while true; do
        running_jobs=$(jobs -r | wc -l)
        if (( running_jobs < MAX_JOBS )); then
            free_gpu=$(get_free_gpu)
            if [ -n "$free_gpu" ]; then
                echo $free_gpu
                return
            fi
        fi
        sleep $CHECK_INTERVAL
    done
}

# 任务提交函数
submit_task() {
    local method=$1
    local N=$2
    local LAMBDA=$3
    local gpu_id=$(wait_for_free_gpu)
    local log_file

    if [[ "$method" == "base" ]]; then
        log_file=${LOG_DIR}/base_N${N}_${MODEL}_${DATASET}.log
        echo "[GPU $gpu_id] 启动 base.py (N=${N})"
        CUDA_VISIBLE_DEVICES=$gpu_id python base.py \
            --n_repetitive_sampling $N --subset_size 1 \
            > "$log_file" 2>&1 &
        return
    fi
    
    if [ -z "$LAMBDA" ]; then
        log_file=${LOG_DIR}/${method}_N${N}_${MODEL}_${DATASET}.log
        echo "[GPU $gpu_id] 启动 ${method} (N=${N})"
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
            --method $method --n_repetitive_sampling $N --subset_size 1 \
            > "$log_file" 2>&1 &
    else
        log_file=${LOG_DIR}/${method}_N${N}_lambda${LAMBDA}_${MODEL}_${DATASET}.log
        echo "[GPU $gpu_id] 启动 ${method} (N=${N}, λ=${LAMBDA})"
        CUDA_VISIBLE_DEVICES=$gpu_id python main.py \
            --method $method --n_repetitive_sampling $N --lambda_weight $LAMBDA --subset_size 1 \
            > "$log_file" 2>&1 &
    fi
    
}

# baseline（只跑一次）
submit_task "baseline" 1 ""

# self-consistency
for N in "${SAMPLE_SIZES[@]}"; do
    submit_task "self-consistency" $N ""
    submit_task "base" $N ""
done

# 其余方法（N × λ）
for method in "entropy" "self-certainty" "self-eval" "coe-c"; do
    for N in "${SAMPLE_SIZES[@]}"; do
        for LAMBDA in "${LAMBDA_WEIGHTS[@]}"; do
            submit_task "$method" $N $LAMBDA
        done
    done
done

# 等所有任务完成
wait
echo "✅ 所有任务完成！"
echo "日志保存在：$LOG_DIR"
echo "输出保存在：$OUTPUT_DIR"
