#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path
from collections import Counter

# 경로 설정
jsonl_path = "/home/intern/banner_vis/data/crop_7000/minimal_7000.jsonl"
target_dir = "/home/intern/banner_vis/data/crop_7000/images"
source_dir = "/home/intern/cropped-banner/image"

def analyze_jsonl():
    """JSONL 파일을 분석하여 중복된 이미지들을 찾습니다."""
    print("=== JSONL 파일 분석 중 ===")
    
    # JSONL에서 모든 이미지 이름과 라인 정보 수집
    image_lines = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                data = json.loads(line.strip())
                image_name = data.get('image', '')
                if image_name:
                    image_lines.append((image_name, line_num, line.strip()))
    
    print(f"총 라인 수: {len(image_lines)}")
    
    # 중복된 이미지 찾기
    image_counter = Counter([img for img, _, _ in image_lines])
    duplicates = {img: count for img, count in image_counter.items() if count > 1}
    
    print(f"고유한 이미지 수: {len(image_counter)}")
    print(f"중복된 이미지 수: {len(duplicates)}")
    
    return image_lines, image_counter, duplicates

def remove_duplicates_and_fill():
    """중복된 JSON을 제거하고 7000장을 채웁니다."""
    print("\n=== 중복 제거 및 7000장 채우기 시작 ===")
    
    # 1. JSONL 분석
    image_lines, image_counter, duplicates = analyze_jsonl()
    
    # 2. 중복 제거된 JSONL 생성
    unique_images = set()
    new_jsonl_lines = []
    
    for image_name, line_num, line_content in image_lines:
        if image_name not in unique_images:
            unique_images.add(image_name)
            new_jsonl_lines.append(line_content)
        else:
            print(f"중복 제거: {image_name} (라인 {line_num})")
    
    print(f"중복 제거 후 라인 수: {len(new_jsonl_lines)}")
    
    # 3. 현재 이미지 폴더 확인
    existing_images = set()
    if os.path.exists(target_dir):
        for file in os.listdir(target_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                existing_images.add(file)
    
    print(f"현재 이미지 폴더에 있는 이미지 수: {len(existing_images)}")
    
    # 4. 추가로 필요한 이미지 수 계산
    needed_count = 7000 - len(new_jsonl_lines)
    print(f"추가로 필요한 이미지 수: {needed_count}")
    
    if needed_count <= 0:
        print("이미 7000개 이상입니다!")
        return
    
    # 5. cropped-banner에서 사용 가능한 이미지들 찾기
    available_images = set()
    if os.path.exists(source_dir):
        for file in os.listdir(source_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                available_images.add(file)
    
    print(f"cropped-banner에서 사용 가능한 이미지 수: {len(available_images)}")
    
    # 6. 아직 사용되지 않은 이미지들 찾기
    unused_images = available_images - existing_images - unique_images
    print(f"아직 사용되지 않은 이미지 수: {len(unused_images)}")
    
    # 7. 추가할 이미지들 선택
    images_to_add = list(unused_images)[:needed_count]
    print(f"추가할 이미지 수: {len(images_to_add)}")
    
    # 8. 새로운 JSONL 파일 생성
    backup_path = jsonl_path + ".backup"
    print(f"기존 JSONL 백업: {backup_path}")
    shutil.copy2(jsonl_path, backup_path)
    
    # 새로운 JSONL 작성
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        # 기존 중복 제거된 라인들
        for line in new_jsonl_lines:
            f.write(line + '\n')
        
        # 새로운 이미지들 추가
        for i, image_name in enumerate(images_to_add):
            new_entry = {
                "image": image_name,
                "added_for_fill": True,
                "fill_index": i + 1
            }
            f.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
    
    print(f"새로운 JSONL 파일 생성 완료: {len(new_jsonl_lines) + len(images_to_add)}개 라인")
    
    # 9. 새로운 이미지들 복사
    print("\n=== 새로운 이미지들 복사 중 ===")
    copied_count = 0
    for image_name in images_to_add:
        src_path = os.path.join(source_dir, image_name)
        dst_path = os.path.join(target_dir, image_name)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            print(f"복사됨: {image_name}")
        else:
            print(f"Warning: 소스에 없음: {image_name}")
    
    print(f"\n=== 완료 ===")
    print(f"복사된 이미지: {copied_count}개")
    
    # 10. 최종 확인
    final_count = len([f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"최종 이미지 수: {final_count}개")
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        final_lines = sum(1 for line in f if line.strip())
    print(f"최종 JSONL 라인 수: {final_lines}개")

def main():
    remove_duplicates_and_fill()

if __name__ == "__main__":
    main() 