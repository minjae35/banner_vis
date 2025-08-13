#!/bin/bash

# Qwen2.5-VL Simple Classification Fine-tuning Script
# CW-Only Simple Classification Experiment

# Required variables
EXPERIMENT_NAME="cw_only_simple_class"
DATASET_CONFIG="cw_only_simple_class%100"
GPU_IDS="0"  # 단일 GPU 사용 (실제로는 CUDA_VISIBLE_DEVICES로 오버라이드됨)
NUM_GPUS=1   # 단일 GPU 사용
EPOCHS=20
LEARNING_RATE=5e-6
BATCH_SIZE=4
GRAD_ACCUM_STEPS=4
USE_SIMPLE="true"  # Simple classification 사용

# GPU 설정
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# CPU 스레드 설정 (OMP 경고 해결)
export OMP_NUM_THREADS=1

# 실험 이름에 따른 접미사 설정
if [ "$USE_SIMPLE" = "true" ]; then
    SUFFIX="_simple_class"
    STRUCTURE_DESC="이미지 → 바로 최종 분류 (텍스트 추출 없음)"
else
    SUFFIX=""
    STRUCTURE_DESC="Chain of Thought (CoT) 사용"
fi

# 로그 파일 설정
log_file="train_${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NUM_GPUS}

# DeepSpeed configuration
deepspeed=./scripts/configs/zero3_large_batch.json

# Model configuration
llm=/home/intern/banner_vis/pretrained/Qwen2.5-VL-3B-Instruct

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Output configuration
run_name="qwen2vl-${EXPERIMENT_NAME}-experiment-gpu${GPU_IDS//,/}"
output_dir=./checkpoints_${EXPERIMENT_NAME}_experiment_gpu${GPU_IDS//,/}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${DATASET_CONFIG} \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --bf16 \
    --output_dir ${output_dir} \
    --cache_dir ./cache \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
    --max_pixels 451584 \
    --min_pixels 451584 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --learning_rate ${LEARNING_RATE} \
    --mm_projector_lr 2e-5 \
    --vision_tower_lr 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --warmup_steps 1000 \
    --max_grad_norm 1 \
    --lr_scheduler_type "linear" \
    --logging_steps 10 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to none"

echo "Starting ${EXPERIMENT_NAME} experiment with GPU ${GPU_IDS}..."
echo "Log file: ${log_file}"
echo "Output directory: ${output_dir}"
echo "GPUs: ${GPU_IDS} (${NUM_GPUS}개)"
echo "Dataset: ${DATASET_CONFIG}"
echo "Structure: ${STRUCTURE_DESC}"
echo "Epochs: ${EPOCHS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Batch Size: ${BATCH_SIZE} (GPU당) × ${NUM_GPUS} (GPU) × ${GRAD_ACCUM_STEPS} (누적) = $((BATCH_SIZE * NUM_GPUS * GRAD_ACCUM_STEPS)) (실제)"
echo "Expected time: 8-10 hours (더 빠름)"

# Launch distributed training with log file
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} 2>&1 | tee ${log_file} 