#!/bin/bash

# CW-Only 실험 실행 스크립트
# Crop + Warp만 사용: Crop 2,800 + Warp 2,800 = 5,600개

# 실험 설정
export EXPERIMENT_NAME="cw_only"
export DATASET_CONFIG="cw_only%100"
export GPU_IDS="2,3"
export NUM_GPUS=2
export EPOCHS=30
export LEARNING_RATE=5e-6
export BATCH_SIZE=2
export GRAD_ACCUM_STEPS=4
export USE_SIMPLE="false"

# 템플릿 스크립트 실행
source ./scripts/templates/training_template.sh 