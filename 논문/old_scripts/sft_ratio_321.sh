#!/bin/bash

# GPU 설정 - 2개 GPU 사용 (6, 7번)
export CUDA_VISIBLE_DEVICES=6,7

# CPU 스레드 설정 (OMP 경고 해결)
export OMP_NUM_THREADS=1

# 로그 파일 설정
log_file="train_c3f2w1_$(date +%Y%m%d_%H%M%S).log"

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 50001-59999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=2  # GPU 6, 7 사용 (2개)

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

# Dataset configuration - 실험 4: C3F2W1 (Crop 3: Flat 2: Warp 1 비율)
datasets="c3f2w1%100"

# Output configuration
run_name="qwen2vl-c3f2w1-experiment-gpu67"
output_dir=./checkpoints_c3f2w1_experiment_gpu67

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
    --num_train_epochs 20 \
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

echo "Starting C3F2W1 experiment (Crop 3: Flat 2: Warp 1 비율) with GPU 6,7..."
echo "Log file: ${log_file}"
echo "Output directory: ${output_dir}"
echo "GPUs: 6, 7 (2개)"
echo "Dataset: c3f2w1 (Crop 4,200 + Flat 2,800 + Warp 1,400 = 8,400개)"
echo "Epochs: 20 (총 262 steps per epoch)"
echo "Expected time: 12-14 hours"

# Launch distributed training with log file
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args} 2>&1 | tee ${log_file} 