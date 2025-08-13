#!/bin/bash

# 간단한 병렬 실행 스크립트
echo "🚀 4개 실험 병렬 실행"
echo "======================"

# 현재 시간
echo "시작: $(date)"
echo ""

# 모든 실험을 백그라운드에서 동시 실행
echo "🔄 모든 실험 시작..."

# BAL-Equal (GPU 0,1)
./scripts/sft_bal_equal.sh > train_bal_equal_$(date +%Y%m%d_%H%M%S).log 2>&1 &
BAL_PID=$!

# CW-Only (GPU 2,3)
./scripts/sft_cw_only.sh > train_cw_only_$(date +%Y%m%d_%H%M%S).log 2>&1 &
CW_PID=$!

# No-Warp (GPU 4,5)
./scripts/sft_no_warp.sh > train_no_warp_$(date +%Y%m%d_%H%M%S).log 2>&1 &
NOWARP_PID=$!

# C3F2W1 (GPU 6,7)
./scripts/sft_ratio_321.sh > train_c3f2w1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
C3F2W1_PID=$!

echo "✅ 모든 실험 시작됨!"
echo ""
echo "📊 프로세스 ID:"
echo "BAL-Equal: $BAL_PID"
echo "CW-Only: $CW_PID"
echo "No-Warp: $NOWARP_PID"
echo "C3F2W1: $C3F2W1_PID"
echo ""

# 프로세스 상태 확인
echo "📈 실행 중인 프로세스:"
ps aux | grep -E "(sft_bal_equal|sft_cw_only|sft_no_warp|sft_ratio_321)" | grep -v grep
echo ""

echo "💡 모니터링 명령어:"
echo "  nvidia-smi                    # GPU 사용량"
echo "  ps aux | grep torchrun        # 프로세스 상태"
echo "  tail -f train_bal_equal_*.log # BAL-Equal 로그"
echo "  tail -f train_cw_only_*.log   # CW-Only 로그"
echo "  tail -f train_no_warp_*.log   # No-Warp 로그"
echo "  tail -f train_c3f2w1_*.log    # C3F2W1 로그"
echo ""
echo "🛑 종료 명령어:"
echo "  kill $BAL_PID $CW_PID $NOWARP_PID $C3F2W1_PID"
echo ""

echo "🎉 병렬 실행 완료! 모든 실험이 백그라운드에서 실행 중입니다." 