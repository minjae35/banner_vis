#!/bin/bash

# simple_classification 4개 실험을 각기 다른 로그로 백그라운드 실행 + PID 관리 + 모니터링 안내

echo "🚀 4개 simple_classification 실험 동시 실행 시작"
echo "=================================="
echo "시작 시간: $(date)"
echo ""

# 실험 설정 배열: 이름:스크립트명
declare -a experiments=(
    "bal_equal:sft_bal_equal_simple_class.sh"
    "cw_only:sft_cw_only_simple_class.sh"
    "no_warp:sft_no_warp_simple_class.sh"
    "c3f2w1:sft_c3f2w1_simple_class.sh"
)

# 각 실험을 백그라운드로 실행
for exp in "${experiments[@]}"; do
    IFS=':' read -r exp_name script_name <<< "$exp"
    log_file="train_${exp_name}_simple_class_$(date +%Y%m%d_%H%M%S).log"
    echo "🔄 ${exp_name} 실험 시작: ${script_name} (로그: $log_file)"
    bash "$(dirname "$0")/../experiments/simple_classification/${script_name}" > "$log_file" 2>&1 &
    eval "${exp_name^^}_PID=\$!"
    echo "${exp_name} PID: ${!exp_name^^}_PID"
    sleep 3
done

echo ""
echo "✅ 모든 simple_classification 실험이 백그라운드에서 실행 중입니다."
echo "로그 파일을 확인하려면:"
for exp in "${experiments[@]}"; do
    IFS=':' read -r exp_name script_name <<< "$exp"
    echo "  tail -f train_${exp_name}_simple_class_*.log"
done
echo ""
echo "📊 현재 실행 중인 실험들:"
ps aux | grep -E "sft_.*_simple_class.sh" | grep -v grep
echo ""
echo "실행 중인 프로세스 ID들:"
for exp in "${experiments[@]}"; do
    IFS=':' read -r exp_name script_name <<< "$exp"
    echo "${exp_name}: ${!exp_name^^}_PID"
done
echo ""
kill_cmd="kill"
for exp in "${experiments[@]}"; do
    IFS=':' read -r exp_name script_name <<< "$exp"
    kill_cmd="$kill_cmd \${${exp_name^^}_PID}"
done
echo "프로세스를 종료하려면: $kill_cmd"
echo ""
echo "종료 시간: $(date)"
echo "💡 모니터링 팁:"
echo "  - GPU 사용량: nvidia-smi"
echo "  - 프로세스 상태: ps aux | grep torchrun"
echo "  - 로그 실시간 확인: tail -f train_*.log" 