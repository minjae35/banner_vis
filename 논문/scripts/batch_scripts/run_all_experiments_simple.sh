#!/bin/bash

# 모든 Simple 실험 동시 실행 스크립트 (CoT 없음)
echo "🚀 4개 Simple 실험 동시 실행 시작 (CoT 없음)"
echo "=================================="

# 현재 시간 출력
echo "시작 시간: $(date)"
echo ""

# 실험 설정 배열
declare -a experiments=(
    "bal_equal:0,1:20"
    "cw_only:2,3:30"
    "no_warp:4,5:30"
    "ratio_321:6,7:20"
)

# 1단계: 실험 1, 2 동시 실행 (GPU 0,1 & 2,3)
echo "📋 1단계: BAL-Equal Simple (GPU 0,1) + CW-Only Simple (GPU 2,3) 동시 실행"
echo "BAL-Equal Simple: Crop 2,800 + Flat 2,800 + Warp 2,800 = 8,400개 (20 에포크)"
echo "CW-Only Simple: Crop 2,800 + Warp 2,800 = 5,600개 (30 에포크)"
echo ""

# 실험 1, 2 실행
for i in {0..1}; do
    IFS=':' read -r exp_name gpu_ids epochs <<< "${experiments[$i]}"
    
    echo "🔄 ${exp_name} Simple 실험 시작 (GPU ${gpu_ids})..."
    bash "$(dirname "$0")/../experiments/simple/run_${exp_name}.sh" > train_${exp_name}_simple_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    eval "${exp_name^^}_SIMPLE_PID=\$!"
    echo "${exp_name} Simple PID: ${!exp_name^^}_SIMPLE_PID"
    
    sleep 5
done

echo ""
echo "✅ 1단계 Simple 실험들이 백그라운드에서 실행 중입니다."
echo "로그 파일을 확인하려면:"
echo "  tail -f train_bal_equal_simple_*.log"
echo "  tail -f train_cw_only_simple_*.log"
echo ""

# 프로세스 상태 확인
echo "📊 현재 실행 중인 Simple 실험들:"
ps aux | grep -E "(run_bal_equal_simple|run_cw_only_simple)" | grep -v grep
echo ""

# 2단계 실행 여부 확인
echo "2단계 Simple 실험 (No-Warp + Ratio 3:2:1)을 지금 실행하시겠습니까? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo ""
    echo "📋 2단계: No-Warp Simple (GPU 4,5) + Ratio 3:2:1 Simple (GPU 6,7) 동시 실행"
    echo "No-Warp Simple: Crop 2,800 + Flat 2,800 = 5,600개 (30 에포크)"
    echo "Ratio 3:2:1 Simple: Crop 4,200 + Flat 2,800 + Warp 1,400 = 8,400개 (20 에포크)"
    echo ""

    # 실험 3, 4 실행
    for i in {2..3}; do
        IFS=':' read -r exp_name gpu_ids epochs <<< "${experiments[$i]}"
        
        echo "🔄 ${exp_name} Simple 실험 시작 (GPU ${gpu_ids})..."
        bash "$(dirname "$0")/../experiments/simple/run_${exp_name}.sh" > train_${exp_name}_simple_$(date +%Y%m%d_%H%M%S).log 2>&1 &
        eval "${exp_name^^}_SIMPLE_PID=\$!"
        echo "${exp_name} Simple PID: ${!exp_name^^}_SIMPLE_PID"
        
        sleep 5
    done

    echo ""
    echo "✅ 모든 Simple 실험이 백그라운드에서 실행 중입니다."
    echo "로그 파일을 확인하려면:"
    for exp in "${experiments[@]}"; do
        IFS=':' read -r exp_name gpu_ids epochs <<< "$exp"
        echo "  tail -f train_${exp_name}_simple_*.log"
    done
    echo ""

    # 모든 프로세스 상태 확인
    echo "📊 현재 실행 중인 모든 Simple 실험들:"
    ps aux | grep -E "(run_bal_equal_simple|run_cw_only_simple|run_no_warp_simple|run_ratio_321_simple)" | grep -v grep
    echo ""

    # 프로세스 ID 저장
    echo "실행 중인 Simple 프로세스 ID들:"
    for exp in "${experiments[@]}"; do
        IFS=':' read -r exp_name gpu_ids epochs <<< "$exp"
        echo "${exp_name} Simple: ${!exp_name^^}_SIMPLE_PID"
    done
    echo ""
    
    # 종료 명령어 생성
    kill_cmd="kill"
    for exp in "${experiments[@]}"; do
        IFS=':' read -r exp_name gpu_ids epochs <<< "$exp"
        kill_cmd="$kill_cmd \${${exp_name^^}_SIMPLE_PID}"
    done
    echo "프로세스를 종료하려면: $kill_cmd"
    echo ""

else
    echo ""
    echo "2단계 Simple 실험은 나중에 수동으로 실행하세요:"
    echo "  ./scripts/experiments/simple/run_no_warp.sh"
    echo "  ./scripts/experiments/simple/run_ratio_321.sh"
    echo ""
    echo "현재 실행 중인 Simple 프로세스 ID들:"
    for i in {0..1}; do
        IFS=':' read -r exp_name gpu_ids epochs <<< "${experiments[$i]}"
        echo "${exp_name} Simple: ${!exp_name^^}_SIMPLE_PID"
    done
    echo ""
    
    kill_cmd="kill"
    for i in {0..1}; do
        IFS=':' read -r exp_name gpu_ids epochs <<< "${experiments[$i]}"
        kill_cmd="$kill_cmd \${${exp_name^^}_SIMPLE_PID}"
    done
    echo "프로세스를 종료하려면: $kill_cmd"
    echo ""
fi

echo "🎉 Simple 실험 실행 완료!"
echo "종료 시간: $(date)"
echo ""
echo "💡 모니터링 팁:"
echo "  - GPU 사용량: nvidia-smi"
echo "  - 프로세스 상태: ps aux | grep torchrun"
echo "  - 로그 실시간 확인: tail -f train_*_simple_*.log" 