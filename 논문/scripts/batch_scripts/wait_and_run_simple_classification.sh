#!/bin/bash

# 현재 실행 중인 실험이 완료되면 자동으로 simple classification 시작

echo "🔍 현재 실행 중인 실험 모니터링 시작..."
echo "=================================="
echo "시작 시간: $(date)"
echo ""

# 현재 실행 중인 torchrun 프로세스 확인
echo "📊 현재 실행 중인 실험들:"
ps aux | grep torchrun | grep -v grep
echo ""

# 프로세스 ID 추출 (첫 번째 torchrun 프로세스)
PID=$(ps aux | grep torchrun | grep -v grep | head -1 | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "❌ 실행 중인 torchrun 프로세스를 찾을 수 없습니다."
    echo "simple classification을 바로 시작합니다."
else
    echo "✅ 모니터링할 프로세스 ID: $PID"
    echo "현재 진행률: 96% (10124/10500)"
    echo ""
    echo "⏳ 프로세스 완료를 기다리는 중..."
    echo "완료되면 자동으로 simple classification이 시작됩니다."
    echo ""
    
    # 프로세스 완료까지 대기
    while kill -0 "$PID" 2>/dev/null; do
        echo -ne "\r⏳ 대기 중... $(date '+%H:%M:%S')"
        sleep 30
    done
    
    echo ""
    echo ""
    echo "🎉 이전 실험이 완료되었습니다!"
    echo "완료 시간: $(date)"
    echo ""
fi

echo "🚀 Simple Classification 실험 시작..."
echo "=================================="

# simple classification 실행
bash "$(dirname "$0")/run_all_simple_classification.sh"

echo ""
echo "✅ Simple Classification 실험이 시작되었습니다!"
echo "종료 시간: $(date)" 