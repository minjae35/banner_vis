#!/usr/bin/env python3
import json
import os
import shutil
import random

def read_jsonl(file_path):
    """JSONL 파일 읽기"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def main():
    # 경로 설정
    jsonl_path = "/home/intern/banner_vis/data/crop_7000/minimal_7000.jsonl"
    source_images = "/home/intern/cropped-banner/image"
    target_images = "/home/intern/banner_vis/data/crop_7000/images"
    
    print("=== 누락된 46개 이미지 추가 ===")
    
    # 1. JSONL에서 이미지 이름 추출
    print("JSONL에서 이미지 이름 추출 중...")
    data = read_jsonl(jsonl_path)
    jsonl_images = set()
    for item in data:
        image_name = item.get('image')
        if image_name:
            jsonl_images.add(image_name)
    
    print(f"JSONL의 이미지 수: {len(jsonl_images)}")
    
    # 2. 현재 존재하는 이미지 확인
    print("현재 존재하는 이미지 확인 중...")
    existing_images = set()
    for filename in os.listdir(target_images):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            existing_images.add(filename)
    
    print(f"현재 존재하는 이미지 수: {len(existing_images)}")
    
    # 3. 누락된 이미지 찾기
    missing_images = jsonl_images - existing_images
    print(f"누락된 이미지 수: {len(missing_images)}")
    
    # 4. 누락된 이미지 복사
    print("누락된 이미지 복사 중...")
    copied_count = 0
    for image_name in missing_images:
        src_path = os.path.join(source_images, image_name)
        dst_path = os.path.join(target_images, image_name)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            print(f"복사됨: {image_name}")
        else:
            print(f"Warning: 소스에 이미지가 없습니다: {image_name}")
    
    print(f"\n=== 완료 ===")
    print(f"복사된 이미지: {copied_count}개")
    
    # 5. 최종 확인
    final_count = len(os.listdir(target_images))
    print(f"최종 이미지 수: {final_count}개")

if __name__ == "__main__":
    main() 