#!/bin/bash

# 모든 실험 동시 실행 스크립트 (개선된 버전)
echo "🚀 4개 실험 동시 실행 시작"
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
echo "📋 1단계: BAL-Equal (GPU 0,1) + CW-Only (GPU 2,3) 동시 실행"
echo "BAL-Equal: Crop 2,800 + Flat 2,800 + Warp 2,800 = 8,400개 (20 에포크)"
echo "CW-Only: Crop 2,800 + Warp 2,800 = 5,600개 (30 에포크)"
echo ""

# 실험 1, 2 실행
for i in {0..1}; do
    IFS=':' read -r exp_name gpu_ids epochs <<< "${experiments[$i]}"
    
    echo "🔄 ${exp_name} 실험 시작 (GPU ${gpu_ids})..."
    ./scripts/experiments/cot/run_${exp_name}.sh > train_${exp_name}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    eval "${exp_name^^}_PID=\$!"
    echo "${exp_name} PID: ${!exp_name^^}_PID"
    
    sleep 5
done

echo ""
echo "✅ 1단계 실험들이 백그라운드에서 실행 중입니다."
echo "로그 파일을 확인하려면:"
echo "  tail -f train_bal_equal_*.log"
echo "  tail -f train_cw_only_*.log"
echo ""

# 프로세스 상태 확인
echo "📊 현재 실행 중인 실험들:"
ps aux | grep -E "(run_bal_equal|run_cw_only)" | grep -v grep
echo ""

# 2단계 실행 여부 확인
echo "2단계 실험 (No-Warp + Ratio 3:2:1)을 지금 실행하시겠습니까? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo ""
    echo "📋 2단계: No-Warp (GPU 4,5) + Ratio 3:2:1 (GPU 6,7) 동시 실행"
    echo "No-Warp: Crop 2,800 + Flat 2,800 = 5,600개 (30 에포크)"
    echo "Ratio 3:2:1: Crop 4,200 + Flat 2,800 + Warp 1,400 = 8,400개 (20 에포크)"
    echo ""

    # 실험 3, 4 실행
    for i in {2..3}; do
        IFS=':' read -r exp_name gpu_ids epochs <<< "${experiments[$i]}"
        
            echo "🔄 ${exp_name} 실험 시작 (GPU ${gpu_ids})..."
    ./scripts/experiments/cot/run_${exp_name}.sh > train_${exp_name}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
        eval "${exp_name^^}_PID=\$!"
        echo "${exp_name} PID: ${!exp_name^^}_PID"
        
        sleep 5
    done

    echo ""
    echo "✅ 모든 실험이 백그라운드에서 실행 중입니다."
    echo "로그 파일을 확인하려면:"
    for exp in "${experiments[@]}"; do
        IFS=':' read -r exp_name gpu_ids epochs <<< "$exp"
        echo "  tail -f train_${exp_name}_*.log"
    done
    echo ""

    # 모든 프로세스 상태 확인
    echo "📊 현재 실행 중인 모든 실험들:"
    ps aux | grep -E "(run_bal_equal|run_cw_only|run_no_warp|run_ratio_321)" | grep -v grep
    echo ""

    # 프로세스 ID 저장
    echo "실행 중인 프로세스 ID들:"
    for exp in "${experiments[@]}"; do
        IFS=':' read -r exp_name gpu_ids epochs <<< "$exp"
        echo "${exp_name}: ${!exp_name^^}_PID"
    done
    echo ""
    
    # 종료 명령어 생성
    kill_cmd="kill"
    for exp in "${experiments[@]}"; do
        IFS=':' read -r exp_name gpu_ids epochs <<< "$exp"
        kill_cmd="$kill_cmd \${${exp_name^^}_PID}"
    done
    echo "프로세스를 종료하려면: $kill_cmd"
    echo ""

else
    echo ""
    echo "2단계 실험은 나중에 수동으로 실행하세요:"
    echo "  ./scripts/experiments/cot/run_no_warp.sh"
    echo "  ./scripts/experiments/cot/run_ratio_321.sh"
    echo ""
    echo "현재 실행 중인 프로세스 ID들:"
    for i in {0..1}; do
        IFS=':' read -r exp_name gpu_ids epochs <<< "${experiments[$i]}"
        echo "${exp_name}: ${!exp_name^^}_PID"
    done
    echo ""
    
    kill_cmd="kill"
    for i in {0..1}; do
        IFS=':' read -r exp_name gpu_ids epochs <<< "${experiments[$i]}"
        kill_cmd="$kill_cmd \${${exp_name^^}_PID}"
    done
    echo "프로세스를 종료하려면: $kill_cmd"
    echo ""
fi

echo "🎉 실험 실행 완료!"
echo "종료 시간: $(date)"
echo ""
echo "💡 모니터링 팁:"
echo "  - GPU 사용량: nvidia-smi"
echo "  - 프로세스 상태: ps aux | grep torchrun"
echo "  - 로그 실시간 확인: tail -f train_*.log" 