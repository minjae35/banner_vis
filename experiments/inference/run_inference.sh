#!/bin/bash


# < 추론 스크립트 예시들 (복붙용) >
echo "=== 추론 스크립트 예시들 ==="


# < Sample 2 >
# data_file, prompt_type, single_checkpoint
echo "Setting data_file, prompt_type, single_checkpoint"
# GPU 설정
export CUDA_VISIBLE_DEVICES=0,1,2,3
python run_inference.py \
    --prompt_type "standard" \
    --single_checkpoint "bal_equal_linebreak_full"
echo -e "\n" && echo "="*50 && echo -e "\n"



