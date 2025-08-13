#!/bin/bash

# simple_classification 4개 실험을 GPU 4,5,6,7번에 각각 하나씩 분산 실행

echo "🚀 4개 simple_classification 실험을 GPU 4,5,6,7번에 분산 실행"
echo "=================================="
echo "시작 시간: $(date)"
echo ""

# 실험 설정 배열: 이름:스크립트명:GPU_ID
declare -a experiments=(
    "bal_equal:sft_bal_equal_simple_class.sh:4"
    "cw_only:sft_cw_only_simple_class.sh:5"
    "no_warp:sft_no_warp_simple_class.sh:6"
    "c3f2w1:sft_c3f2w1_simple_class.sh:7"
)

# 각 실험을 지정된 GPU에서 백그라운드로 실행
declare -A pids
for exp in "${experiments[@]}"; do
    IFS=':' read -r exp_name script_name gpu_id <<< "$exp"
    log_file="train_${exp_name}_simple_class_$(date +%Y%m%d_%H%M%S).log"
    echo "🔄 ${exp_name} 실험 시작: ${script_name} (GPU ${gpu_id}, 로그: $log_file)"
    
    # GPU ID를 환경변수로 설정하고 스크립트 실행
    CUDA_VISIBLE_DEVICES=${gpu_id} bash "$(dirname "$0")/../experiments/simple_classification/${script_name}" > "$log_file" 2>&1 &
    pids[$exp_name]=$!
    echo "${exp_name} PID: ${pids[$exp_name]} (GPU ${gpu_id})"
    sleep 3
done

echo ""
echo "✅ 모든 simple_classification 실험이 각 GPU에서 백그라운드로 실행 중입니다."
echo "GPU 할당:"
echo "  - BAL-Equal: GPU 4"
echo "  - CW-Only: GPU 5"
echo "  - No-Warp: GPU 6"
echo "  - C3F2W1: GPU 7"
echo ""
echo "로그 파일을 확인하려면:"
for exp in "${experiments[@]}"; do
    IFS=':' read -r exp_name script_name gpu_id <<< "$exp"
    echo "  tail -f train_${exp_name}_simple_class_*.log"
done
echo ""
echo "📊 현재 실행 중인 실험들:"
ps aux | grep -E "sft_.*_simple_class.sh" | grep -v grep
echo ""
echo "실행 중인 프로세스 ID들:"
for exp in "${experiments[@]}"; do
    IFS=':' read -r exp_name script_name gpu_id <<< "$exp"
    echo "${exp_name}: ${pids[$exp_name]} (GPU ${gpu_id})"
done
echo ""
kill_cmd="kill"
for exp in "${experiments[@]}"; do
    IFS=':' read -r exp_name script_name gpu_id <<< "$exp"
    kill_cmd="$kill_cmd ${pids[$exp_name]}"
done
echo "프로세스를 종료하려면: $kill_cmd"
echo ""
echo "종료 시간: $(date)"
echo "💡 모니터링 팁:"
echo "  - GPU 사용량: nvidia-smi"
echo "  - 프로세스 상태: ps aux | grep torchrun"
echo "  - 로그 실시간 확인: tail -f train_*.log" 