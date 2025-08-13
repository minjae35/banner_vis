#!/bin/bash

# No-Warp Simple 실험 실행 스크립트
# Crop + Flat만 사용 (CoT 없음): Crop 2,800 + Flat 2,800 = 5,600개

# 실험 설정
export EXPERIMENT_NAME="no_warp"
export DATASET_CONFIG="no_warp%100"
export GPU_IDS="4,5"
export NUM_GPUS=2
export EPOCHS=30
export LEARNING_RATE=5e-6
export BATCH_SIZE=2
export GRAD_ACCUM_STEPS=4
export USE_SIMPLE="true"

# 템플릿 스크립트 실행
source ./scripts/templates/training_template.sh 