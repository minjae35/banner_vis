#!/bin/bash

# 병렬 실험 실행 스크립트
# BAL-Equal과 CW-Only를 GPU 4,5와 6,7에서 동시 실행

echo "🎯 Banner Classification Parallel Experiments"
echo "============================================="
echo "실험 설계: README 기반 논문 수준 실험"
echo "GPU 분배: 4,5번 (BAL-Equal) / 6,7번 (CW-Only)"
echo "예상 시간: 12-14시간 (병렬 실행)"
echo ""

# 시작 시간 기록
start_time=$(date)
echo "🚀 실험 시작 시간: ${start_time}"
echo ""

# 실험 정보
echo "📊 실행할 실험:"
echo "1. BAL-Equal (GPU 4,5): 완전 균형 Baseline"
echo "   - 데이터: Crop 2,800 + Flat 2,800 + Warp 2,800 = 8,400개"
echo "   - Epoch: 20"
echo "   - 출력: checkpoints_bal_equal_experiment_gpu45/"
echo ""
echo "2. CW-Only (GPU 6,7): Flat 없이 자연·기하 왜곡만 학습"
echo "   - 데이터: Crop 2,800 + Warp 2,800 = 5,600개"
echo "   - Epoch: 30"
echo "   - 출력: checkpoints_cw_only_experiment_gpu67/"
echo ""

# 스크립트 실행 권한 확인
chmod +x "./scripts/sft_bal_equal.sh"
chmod +x "./scripts/sft_cw_only.sh"

echo "=========================================="
echo "🚀 병렬 실험 시작..."
echo "=========================================="

# 백그라운드에서 BAL-Equal 실행
echo "📊 BAL-Equal 실험 시작 (GPU 4,5)..."
bash "./scripts/sft_bal_equal.sh" &
BAL_EQUAL_PID=$!
echo "BAL-Equal PID: ${BAL_EQUAL_PID}"

# 잠시 대기
sleep 5

# 백그라운드에서 CW-Only 실행
echo "📊 CW-Only 실험 시작 (GPU 6,7)..."
bash "./scripts/sft_cw_only.sh" &
CW_ONLY_PID=$!
echo "CW-Only PID: ${CW_ONLY_PID}"

echo ""
echo "⏳ 두 실험이 병렬로 실행 중입니다..."
echo "BAL-Equal PID: ${BAL_EQUAL_PID}"
echo "CW-Only PID: ${CW_ONLY_PID}"
echo ""

# 실험 진행 상황 모니터링
echo "📈 실험 진행 상황 모니터링:"
while kill -0 ${BAL_EQUAL_PID} 2>/dev/null || kill -0 ${CW_ONLY_PID} 2>/dev/null; do
    echo "$(date): 실험 진행 중..."
    
    # BAL-Equal 상태 확인
    if kill -0 ${BAL_EQUAL_PID} 2>/dev/null; then
        echo "  ✅ BAL-Equal (GPU 4,5): 실행 중"
    else
        echo "  🏁 BAL-Equal (GPU 4,5): 완료"
    fi
    
    # CW-Only 상태 확인
    if kill -0 ${CW_ONLY_PID} 2>/dev/null; then
        echo "  ✅ CW-Only (GPU 6,7): 실행 중"
    else
        echo "  🏁 CW-Only (GPU 6,7): 완료"
    fi
    
    echo ""
    sleep 300  # 5분마다 상태 확인
done

# 종료 시간 기록
end_time=$(date)
echo "=========================================="
echo "🎉 모든 실험 완료!"
echo "🚀 시작 시간: ${start_time}"
echo "🏁 종료 시간: ${end_time}"
echo "=========================================="

# 결과 요약
echo ""
echo "📊 실험 결과 요약:"
echo "1. BAL-Equal: checkpoints_bal_equal_experiment_gpu45/"
echo "2. CW-Only: checkpoints_cw_only_experiment_gpu67/"
echo ""
echo "📈 다음 단계: 성능 비교 분석"
echo "   - 각 실험의 최종 모델 로드"
echo "   - 고정 Hold-out 데이터로 평가"
echo "   - 타입별 성능 분석"
echo "   - 조합 효과 분석 (CW-Only vs BAL-Equal)" 