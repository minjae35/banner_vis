#!/bin/bash

echo "🚀 Banner Vis 프로젝트 설정을 시작합니다..."

# 1. 필요한 디렉토리 생성
echo "📁 필요한 디렉토리를 생성합니다..."
mkdir -p data/base_data/images
mkdir -p data/base_data/crop_7000
mkdir -p checkpoints
mkdir -p output
mkdir -p model
mkdir -p pretrained

# 2. Python 패키지 설치
echo "📦 Python 패키지를 설치합니다..."
pip install -r requirements.txt

# 3. 샘플 데이터 다운로드 (선택사항)
echo "📥 샘플 데이터를 다운로드하시겠습니까? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "📥 샘플 데이터를 다운로드합니다..."
    # 여기에 샘플 데이터 다운로드 명령어 추가
    echo "⚠️  실제 데이터는 별도로 준비해주세요."
fi

# 4. 모델 다운로드 (선택사항)
echo "🤖 사전 훈련된 모델을 다운로드하시겠습니까? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "🤖 모델을 다운로드합니다..."
    # 여기에 모델 다운로드 명령어 추가
    echo "⚠️  실제 모델은 별도로 준비해주세요."
fi

echo "✅ 설정이 완료되었습니다!"
echo "📖 README.md 파일을 참고하여 사용법을 확인하세요."
