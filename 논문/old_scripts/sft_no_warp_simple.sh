#!/bin/bash

# GPU 설정 - 2개 GPU 사용 (4, 5번)
export CUDA_VISIBLE_DEVICES=4,5

# CPU 스레드 설정 (OMP 경고 해결)
export OMP_NUM_THREADS=1

# 로그 파일 설정
log_file="train_no_warp_simple_$(date +%Y%m%d_%H%M%S).log"

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 40001-49999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=2  # GPU 4, 5 사용 (2개)

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=/home/intern/banner_vis/pretrained/Qwen2.5-VL-3B-Instruct

# Training hyperparameters
lr=5e-6  # 오버피팅 방지를 위해 learning rate 낮춤
batch_size=2  # GPU당 배치 사이즈 (2 GPU 사용)
grad_accum_steps=4  # 실제 배치 사이즈 = 2 * 2 * 4 = 16

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration - 실험 3: No-Warp (기하 왜곡 없이 자연·디지털만) - CoT 없는 버전
datasets="no_warp%100"

# Output configuration
run_name="qwen2vl-no-warp-simple-experiment-gpu45"
output_dir=./checkpoints_no_warp_simple_experiment_gpu45

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --bf16 \
    --output_dir ${output_dir} \
    --cache_dir ./cache \
    --num_train_epochs 30 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 451584 \
    --min_pixels 451584 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --metric_for_best_model "eval_loss" \
    --greater_is_better False \
    --learning_rate ${lr} \
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

echo "Starting No-Warp Simple experiment (CoT 없는 버전) with GPU 4,5..."
echo "Log file: ${log_file}"
echo "Output directory: ${output_dir}"
echo "GPUs: 4, 5 (2개)"
echo "Dataset: no_warp (Crop 2,800 + Flat 2,800 = 5,600개)"
echo "Structure: 텍스트 추출 → 바로 최종 답안 (CoT 없음)"
echo "Epochs: 30 (총 175 steps per epoch)"
echo "Expected time: 12-14 hours"

# Launch distributed training with log file
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} 2>&1 | tee ${log_file} 