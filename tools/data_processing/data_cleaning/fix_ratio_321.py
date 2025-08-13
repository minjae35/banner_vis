#!/usr/bin/env python3
import json
import os

def fix_ratio_321():
    """Ratio-321_fixed.jsonl에서 존재하지 않는 이미지 제거"""
    print("=== Ratio-321_fixed.jsonl 수정 ===")
    
    input_file = "/home/intern/banner_vis/data/experiments/Ratio-321_fixed.jsonl"
    output_file = "/home/intern/banner_vis/data/experiments/Ratio-321_fixed_cleaned.jsonl"
    
    # 이미지 디렉토리들
    image_dirs = [
        "/home/intern/banner_vis/data/base_data/crop_7000/images_resized/",
        "/home/intern/banner_vis/data/base_data/flat_3500/images/",
        "/home/intern/banner_vis/data/base_data/warp_4500/images/"
    ]
    
    # 모든 실제 이미지 파일 목록
    actual_images = set()
    for dir_path in image_dirs:
        if os.path.exists(dir_path):
            for filename in os.listdir(dir_path):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    actual_images.add(filename)
    
    print(f"실제 이미지 파일 총 개수: {len(actual_images)}개")
    
    # 파일 처리
    valid_data = []
    removed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                image_filename = os.path.basename(item.get('image', ''))
                
                if image_filename in actual_images:
                    valid_data.append(item)
                else:
                    removed_count += 1
                    print(f"제거: 라인 {line_num} - {image_filename}")
                    
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류 라인 {line_num}: {e}")
                removed_count += 1
    
    # 수정된 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"원본 데이터: {line_num}개")
    print(f"유효한 데이터: {len(valid_data)}개")
    print(f"제거된 데이터: {removed_count}개")
    print(f"수정된 파일 저장: {output_file}")
    
    # 기존 파일 백업하고 새 파일로 교체
    backup_file = input_file + ".backup"
    os.rename(input_file, backup_file)
    os.rename(output_file, input_file)
    
    print(f"기존 파일 백업: {backup_file}")
    print("✅ Ratio-321_fixed.jsonl 수정 완료!")

if __name__ == "__main__":
    fix_ratio_321() 