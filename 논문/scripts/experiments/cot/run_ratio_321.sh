#!/bin/bash

# Ratio 3:2:1 실험 실행 스크립트
# 3:2:1 비율: Crop 4,200 + Flat 2,800 + Warp 1,400 = 8,400개

# 실험 설정
export EXPERIMENT_NAME="ratio_321"
export DATASET_CONFIG="ratio_321%100"
export GPU_IDS="6,7"
export NUM_GPUS=2
export EPOCHS=20
export LEARNING_RATE=5e-6
export BATCH_SIZE=2
export GRAD_ACCUM_STEPS=4
export USE_SIMPLE="false"

# 템플릿 스크립트 실행
source ./scripts/templates/training_template.sh 