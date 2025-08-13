#!/usr/bin/env python3
import json
import os

def fix_crop_jsonl():
    """Crop JSONL 파일에서 중복 제거하고 이미지 개수에 맞춤"""
    print("=== Crop JSONL 파일 중복 제거 ===")
    
    # 실제 이미지 파일 목록 가져오기
    image_dir = "/home/intern/banner_vis/data/base_data/crop_7000/images_resized/"
    image_files = set()
    
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_files.add(filename)
    
    print(f"실제 이미지 파일 개수: {len(image_files)}개")
    
    # 기존 JSONL 파일에서 중복 제거하고 이미지가 존재하는 것만 필터링
    input_file = "/home/intern/banner_vis/data/base_data/crop_7000/minimal_7000.jsonl"
    output_file = "/home/intern/banner_vis/data/base_data/crop_7000/minimal_7000_fixed.jsonl"
    
    seen_ids = set()
    valid_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            item_id = item.get('id', '')
            image_filename = os.path.basename(item.get('image', ''))
            
            # 중복 제거하고 이미지가 존재하는 것만
            if item_id not in seen_ids and image_filename in image_files:
                seen_ids.add(item_id)
                valid_data.append(item)
    
    print(f"중복 제거 후 유효한 데이터: {len(valid_data)}개")
    
    # 수정된 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"수정된 파일 저장: {output_file}")
    
    # 기존 파일 백업
    backup_file = input_file + ".backup"
    os.rename(input_file, backup_file)
    print(f"기존 파일 백업: {backup_file}")
    
    # 새 파일을 원래 이름으로 이동
    os.rename(output_file, input_file)
    print(f"수정된 파일을 원래 이름으로 이동: {input_file}")
    
    print("✅ Crop JSONL 파일 수정 완료!")

if __name__ == "__main__":
    fix_crop_jsonl() 