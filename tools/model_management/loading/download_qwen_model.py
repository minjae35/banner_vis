#!/usr/bin/env python3
"""
Qwen2.5-VL-3B-Instruct 모델을 ModelScope에서 다운로드하는 스크립트
"""

import os
import sys
from pathlib import Path

def download_qwen_model():
    """ModelScope에서 Qwen2.5-VL-3B-Instruct 모델을 다운로드합니다."""
    
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("❌ modelscope 패키지가 설치되지 않았습니다.")
        print("다음 명령어로 설치해주세요:")
        print("pip install modelscope")
        sys.exit(1)
    
    # 다운로드 경로 설정
    cache_dir = "/home/intern/banner_vis/pretrained/Qwen2.5-VL-3B-Instruct"
    
    # 디렉토리가 없으면 생성
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Qwen2.5-VL-3B-Instruct 모델 다운로드를 시작합니다...")
    print(f"📁 저장 경로: {cache_dir}")
    print("⏳ 다운로드 중... (시간이 오래 걸릴 수 있습니다)")
    
    try:
        # 모델 다운로드
        snapshot_download(
            'qwen/Qwen2.5-VL-3B-Instruct',
            cache_dir=cache_dir
        )
        
        print("✅ 모델 다운로드가 완료되었습니다!")
        print(f"📂 모델 위치: {cache_dir}")
        
        # 다운로드된 파일 확인
        print("\n📋 다운로드된 파일 목록:")
        for file_path in Path(cache_dir).rglob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  - {file_path.name} ({size_mb:.1f} MB)")
        
        print(f"\n🎯 학습 시 사용할 경로:")
        print(f"--model_name_or_path {cache_dir}")
        
    except Exception as e:
        print(f"❌ 다운로드 중 오류가 발생했습니다: {e}")
        print("💡 해결 방법:")
        print("1. 인터넷 연결을 확인해주세요")
        print("2. ModelScope 계정이 필요할 수 있습니다")
        print("3. VPN이나 중국 내 네트워크에서 더 빠를 수 있습니다")
        sys.exit(1)

if __name__ == "__main__":
    download_qwen_model() 