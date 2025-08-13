#!/usr/bin/env python3
import json
import os
import shutil
from collections import Counter

def fix_validation_1000():
    """Validation set을 정확히 1000개로 맞춥니다."""
    
    print("🔧 Validation Set을 1000개로 수정 중...")
    
    # 경로 설정
    validation_path = "/home/intern/banner_vis/data/experiment_datasets/validation"
    base_data_path = "/home/intern/banner_vis/data/base_data"
    
    # 1. 현재 상황 분석
    print("=== 현재 상황 분석 ===")
    
    # JSONL에서 이미지 이름들 추출
    jsonl_entries = []
    with open(os.path.join(validation_path, "validation.jsonl"), 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                jsonl_entries.append(json.loads(line.strip()))
    
    print(f"JSONL 항목 수: {len(jsonl_entries)}")
    
    # 중복 확인
    image_counter = Counter([entry['image'] for entry in jsonl_entries])
    duplicates = {img: count for img, count in image_counter.items() if count > 1}
    
    print(f"고유한 이미지 수: {len(image_counter)}")
    print(f"중복된 이미지: {duplicates}")
    
    # 2. 중복 제거
    print("\n=== 중복 제거 ===")
    
    # 중복된 이미지의 첫 번째 항목만 유지
    seen_images = set()
    unique_entries = []
    
    for entry in jsonl_entries:
        image_name = entry['image']
        if image_name not in seen_images:
            seen_images.add(image_name)
            unique_entries.append(entry)
        else:
            print(f"중복 제거: {image_name}")
    
    print(f"중복 제거 후 항목 수: {len(unique_entries)}")
    
    # 3. 누락된 이미지 1개 추가
    print("\n=== 누락된 이미지 추가 ===")
    
    # 사용 가능한 이미지들 찾기 (crop, flat, warp에서)
    used_images = set(entry['image'] for entry in unique_entries)
    
    # 각 소스에서 사용되지 않은 이미지 찾기
    available_images = []
    
    for data_type in ['crop', 'flat', 'warp']:
        source_path = os.path.join(base_data_path, f"{data_type}_7000" if data_type == 'crop' else f"{data_type}_3500" if data_type == 'flat' else f"{data_type}_4500")
        jsonl_path = os.path.join(source_path, f"minimal_7000.jsonl" if data_type == 'crop' else f"{data_type}_3500.jsonl" if data_type == 'flat' else f"{data_type}_4500.jsonl")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    if data['image'] not in used_images:
                        available_images.append(data)
                        break  # 하나만 추가
    
    if available_images:
        new_entry = available_images[0]
        unique_entries.append(new_entry)
        print(f"추가된 이미지: {new_entry['image']}")
    else:
        print("Warning: 추가할 이미지를 찾을 수 없습니다!")
    
    print(f"최종 항목 수: {len(unique_entries)}")
    
    # 4. 새로운 validation.jsonl 생성
    print("\n=== 새로운 validation.jsonl 생성 ===")
    
    backup_path = os.path.join(validation_path, "validation.jsonl.backup")
    shutil.copy2(os.path.join(validation_path, "validation.jsonl"), backup_path)
    print(f"백업 생성: {backup_path}")
    
    with open(os.path.join(validation_path, "validation.jsonl"), 'w', encoding='utf-8') as f:
        for entry in unique_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print("새로운 validation.jsonl 생성 완료")
    
    # 5. 이미지 파일 확인 및 복사
    print("\n=== 이미지 파일 확인 ===")
    
    images_dir = os.path.join(validation_path, "images")
    existing_images = set()
    for file in os.listdir(images_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            existing_images.add(file)
    
    print(f"현재 이미지 파일 수: {len(existing_images)}")
    
    # 누락된 이미지 파일 복사
    for entry in unique_entries:
        image_name = entry['image']
        if image_name not in existing_images:
            # 소스에서 찾기
            for data_type in ['crop', 'flat', 'warp']:
                source_path = os.path.join(base_data_path, f"{data_type}_7000" if data_type == 'crop' else f"{data_type}_3500" if data_type == 'flat' else f"{data_type}_4500")
                source_images_path = os.path.join(source_path, "images")
                source_file = os.path.join(source_images_path, image_name)
                
                if os.path.exists(source_file):
                    dst_file = os.path.join(images_dir, image_name)
                    shutil.copy2(source_file, dst_file)
                    print(f"이미지 복사: {image_name}")
                    break
    
    # 6. 최종 확인
    print("\n=== 최종 확인 ===")
    
    final_jsonl_count = 0
    with open(os.path.join(validation_path, "validation.jsonl"), 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                final_jsonl_count += 1
    
    final_image_count = len([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"최종 JSONL 항목 수: {final_jsonl_count}")
    print(f"최종 이미지 파일 수: {final_image_count}")
    
    if final_jsonl_count == final_image_count == 1000:
        print("✅ Validation Set이 정확히 1000개로 수정되었습니다!")
    else:
        print("❌ 아직 문제가 있습니다.")

if __name__ == "__main__":
    fix_validation_1000() 