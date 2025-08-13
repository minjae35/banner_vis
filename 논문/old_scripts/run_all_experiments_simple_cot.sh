#!/bin/bash

# CoT 없는 실험 동시 실행 스크립트
echo "🚀 4개 실험 동시 실행 시작 (CoT 없는 버전)"
echo "=================================="

# 현재 시간 출력
echo "시작 시간: $(date)"
echo ""

# 1단계: 실험 1, 2 동시 실행 (GPU 0,1 & 2,3)
echo "📋 1단계: BAL-Equal (GPU 0,1) + CW-Only (GPU 2,3) 동시 실행"
echo "BAL-Equal: Crop 2,800 + Flat 2,800 + Warp 2,800 = 8,400개 (20 에포크)"
echo "CW-Only: Crop 2,800 + Warp 2,800 = 5,600개 (30 에포크)"
echo "구조: 텍스트 추출 → 바로 최종 답안 (CoT 없음)"
echo ""

# 백그라운드에서 BAL-Equal 실행
echo "🔄 BAL-Equal 실험 시작 (GPU 0,1)..."
./scripts/sft_bal_equal_simple.sh > train_bal_equal_simple_$(date +%Y%m%d_%H%M%S).log 2>&1 &
BAL_PID=$!
echo "BAL-Equal PID: $BAL_PID"

# 잠시 대기
sleep 5

# 백그라운드에서 CW-Only 실행
echo "🔄 CW-Only 실험 시작 (GPU 2,3)..."
./scripts/sft_cw_only_simple.sh > train_cw_only_simple_$(date +%Y%m%d_%H%M%S).log 2>&1 &
CW_PID=$!
echo "CW-Only PID: $CW_PID"

echo ""
echo "✅ 1단계 실험들이 백그라운드에서 실행 중입니다."
echo "로그 파일을 확인하려면:"
echo "  tail -f train_bal_equal_simple_*.log"
echo "  tail -f train_cw_only_simple_*.log"
echo ""

# 프로세스 상태 확인
echo "📊 현재 실행 중인 실험들:"
ps aux | grep -E "(sft_bal_equal_simple|sft_cw_only_simple)" | grep -v grep
echo ""

# 2단계 실행 여부 확인
echo "2단계 실험 (No-Warp + C3F2W1)을 지금 실행하시겠습니까? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo ""
    echo "📋 2단계: No-Warp (GPU 4,5) + C3F2W1 (GPU 6,7) 동시 실행"
    echo "No-Warp: Crop 2,800 + Flat 2,800 = 5,600개 (30 에포크)"
    echo "C3F2W1: Crop 4,200 + Flat 2,800 + Warp 1,400 = 8,400개 (20 에포크)"
    echo "구조: 텍스트 추출 → 바로 최종 답안 (CoT 없음)"
    echo ""

    # 백그라운드에서 No-Warp 실행
    echo "🔄 No-Warp 실험 시작 (GPU 4,5)..."
    ./scripts/sft_no_warp_simple.sh > train_no_warp_simple_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    NOWARP_PID=$!
    echo "No-Warp PID: $NOWARP_PID"

    # 잠시 대기
    sleep 5

    # 백그라운드에서 C3F2W1 실행
    echo "🔄 C3F2W1 실험 시작 (GPU 6,7)..."
    ./scripts/sft_ratio_321_simple.sh > train_c3f2w1_simple_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    C3F2W1_PID=$!
    echo "C3F2W1 PID: $C3F2W1_PID"

    echo ""
    echo "✅ 모든 실험이 백그라운드에서 실행 중입니다."
    echo "로그 파일을 확인하려면:"
    echo "  tail -f train_bal_equal_simple_*.log"
    echo "  tail -f train_cw_only_simple_*.log"
    echo "  tail -f train_no_warp_simple_*.log"
    echo "  tail -f train_c3f2w1_simple_*.log"
    echo ""

    # 모든 프로세스 상태 확인
    echo "📊 현재 실행 중인 모든 실험들:"
    ps aux | grep -E "(sft_bal_equal_simple|sft_cw_only_simple|sft_no_warp_simple|sft_ratio_321_simple)" | grep -v grep
    echo ""

    # 프로세스 ID 저장
    echo "실행 중인 프로세스 ID들:"
    echo "BAL-Equal: $BAL_PID"
    echo "CW-Only: $CW_PID"
    echo "No-Warp: $NOWARP_PID"
    echo "C3F2W1: $C3F2W1_PID"
    echo ""
    echo "프로세스를 종료하려면: kill $BAL_PID $CW_PID $NOWARP_PID $C3F2W1_PID"
    echo ""

else
    echo ""
    echo "2단계 실험은 나중에 수동으로 실행하세요:"
    echo "  ./scripts/sft_no_warp_simple.sh"
    echo "  ./scripts/sft_ratio_321_simple.sh"
    echo ""
    echo "현재 실행 중인 프로세스 ID들:"
    echo "BAL-Equal: $BAL_PID"
    echo "CW-Only: $CW_PID"
    echo ""
    echo "프로세스를 종료하려면: kill $BAL_PID $CW_PID"
    echo ""
fi

echo "🎉 CoT 없는 실험 실행 완료!"
echo "종료 시간: $(date)"
echo ""
echo "💡 모니터링 팁:"
echo "  - GPU 사용량: nvidia-smi"
echo "  - 프로세스 상태: ps aux | grep torchrun"
echo "  - 로그 실시간 확인: tail -f train_*_simple_*.log"
echo ""
echo "📁 생성될 체크포인트 디렉토리:"
echo "  - checkpoints_bal_equal_simple_experiment_gpu01/"
echo "  - checkpoints_cw_only_simple_experiment_gpu23/"
echo "  - checkpoints_no_warp_simple_experiment_gpu45/"
echo "  - checkpoints_c3f2w1_simple_experiment_gpu67/" 