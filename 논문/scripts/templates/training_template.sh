#!/bin/bash

# Qwen2.5-VL Fine-tuning Template Script
# Usage: source this script and set the required variables

# Required variables (must be set before sourcing):
# - EXPERIMENT_NAME: 실험 이름 (e.g., "bal_equal", "cw_only")
# - DATASET_CONFIG: 데이터셋 설정 (e.g., "bal_equal%100")
# - GPU_IDS: 사용할 GPU ID들 (e.g., "0,1")
# - NUM_GPUS: GPU 개수 (e.g., 2)
# - EPOCHS: 훈련 에포크 수 (e.g., 20)
# - LEARNING_RATE: 학습률 (e.g., 5e-6)
# - BATCH_SIZE: GPU당 배치 사이즈 (e.g., 2)
# - GRAD_ACCUM_STEPS: 그래디언트 누적 스텝 (e.g., 4)
# - USE_SIMPLE: Simple CoT 사용 여부 (true/false)

# GPU 설정
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

# CPU 스레드 설정 (OMP 경고 해결)
export OMP_NUM_THREADS=1

# 실험 이름에 따른 접미사 설정
if [ "$USE_SIMPLE" = "true" ]; then
    SUFFIX="_simple"
    STRUCTURE_DESC="텍스트 추출 → 바로 최종 답안 (CoT 없음)"
else
    SUFFIX=""
    STRUCTURE_DESC="Chain of Thought (CoT) 사용"
fi

# 로그 파일 설정
log_file="train_${EXPERIMENT_NAME}${SUFFIX}_$(date +%Y%m%d_%H%M%S).log"

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NUM_GPUS}

# DeepSpeed configuration
deepspeed=./scripts/configs/zero3.json

# Model configuration
llm=/home/intern/banner_vis/pretrained/Qwen2.5-VL-3B-Instruct

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Output configuration
run_name="qwen2vl-${EXPERIMENT_NAME}${SUFFIX}-experiment-gpu${GPU_IDS//,/}"
output_dir=./checkpoints_${EXPERIMENT_NAME}${SUFFIX}_experiment_gpu${GPU_IDS//,/}

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


# Launch distributed training with log file
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} 2>&1 | tee ${log_file} 