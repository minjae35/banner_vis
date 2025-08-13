#!/bin/bash

# BAL-Equal 실험 실행 스크립트
# 완전 균형 Baseline: Crop 2,800 + Flat 2,800 + Warp 2,800 = 8,400개

# 실험 설정
export EXPERIMENT_NAME="bal_equal"
export DATASET_CONFIG="bal_equal%100"
export GPU_IDS="0,1"
export NUM_GPUS=2
export EPOCHS=20
export LEARNING_RATE=5e-6
export BATCH_SIZE=2
export GRAD_ACCUM_STEPS=4
export USE_SIMPLE="false"

# 템플릿 스크립트 실행
source ./scripts/templates/training_template.sh 