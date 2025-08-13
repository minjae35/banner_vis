#!/usr/bin/env python3
import json
import os
import random

def add_one_to_ratio_321():
    """Ratio-321_fixed.jsonl에 1개 데이터 추가"""
    print("=== Ratio-321_fixed.jsonl에 1개 추가 ===")
    
    input_file = "/home/intern/banner_vis/data/experiments/Ratio-321_fixed.jsonl"
    output_file = "/home/intern/banner_vis/data/experiments/Ratio-321_fixed_updated.jsonl"
    
    # 기존 데이터 로드
    existing_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            existing_data.append(json.loads(line.strip()))
    
    print(f"현재 데이터 개수: {len(existing_data)}개")
    
    # 사용 가능한 이미지 찾기 (기존 데이터에서 사용되지 않은 것)
    used_images = set()
    for item in existing_data:
        image_filename = os.path.basename(item.get('image', ''))
        used_images.add(image_filename)
    
    # 각 타입별 사용 가능한 이미지 찾기
    available_images = {
        'crop': [],
        'flat': [],
        'warp': []
    }
    
    # Crop 이미지 확인
    crop_dir = "/home/intern/banner_vis/data/base_data/crop_7000/images_resized/"
    if os.path.exists(crop_dir):
        for filename in os.listdir(crop_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')) and filename not in used_images:
                available_images['crop'].append(filename)
    
    # Flat 이미지 확인
    flat_dir = "/home/intern/banner_vis/data/base_data/flat_3500/images/"
    if os.path.exists(flat_dir):
        for filename in os.listdir(flat_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')) and filename not in used_images:
                available_images['flat'].append(filename)
    
    # Warp 이미지 확인
    warp_dir = "/home/intern/banner_vis/data/base_data/warp_4500/images/"
    if os.path.exists(warp_dir):
        for filename in os.listdir(warp_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')) and filename not in used_images:
                available_images['warp'].append(filename)
    
    print(f"사용 가능한 이미지:")
    print(f"  Crop: {len(available_images['crop'])}개")
    print(f"  Flat: {len(available_images['flat'])}개")
    print(f"  Warp: {len(available_images['warp'])}개")
    
    # Ratio-321 비율에 맞춰 추가할 타입 결정 (3:2:1 비율)
    # 현재 데이터 분석
    crop_count = sum(1 for item in existing_data if 'crop' in item.get('image', ''))
    flat_count = sum(1 for item in existing_data if 'flat' in item.get('image', ''))
    warp_count = sum(1 for item in existing_data if 'warp' in item.get('image', ''))
    
    print(f"현재 비율 - Crop: {crop_count}, Flat: {flat_count}, Warp: {warp_count}")
    
    # 목표 비율 (3:2:1)
    target_crop = 4200
    target_flat = 2800
    target_warp = 1400
    
    # 어떤 타입을 추가할지 결정
    if crop_count < target_crop and available_images['crop']:
        add_type = 'crop'
        add_dir = crop_dir
    elif flat_count < target_flat and available_images['flat']:
        add_type = 'flat'
        add_dir = flat_dir
    elif warp_count < target_warp and available_images['warp']:
        add_type = 'warp'
        add_dir = warp_dir
    else:
        print("❌ 추가할 수 있는 이미지가 없습니다.")
        return
    
    # 랜덤하게 이미지 선택
    selected_image = random.choice(available_images[add_type])
    image_path = os.path.join(add_dir, selected_image)
    
    # 새 데이터 생성
    new_item = {
        "id": f"added_{add_type}_{len(existing_data)}",
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": "이 배너 이미지를 분류해주세요."
            },
            {
                "from": "assistant", 
                "value": f"이 배너는 {add_type} 타입입니다."
            }
        ]
    }
    
    # 데이터 추가
    existing_data.append(new_item)
    
    print(f"추가된 데이터: {add_type} 타입 - {selected_image}")
    
    # 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in existing_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"업데이트된 파일 저장: {output_file}")
    
    # 기존 파일 백업하고 새 파일로 교체
    backup_file = input_file + ".backup2"
    os.rename(input_file, backup_file)
    os.rename(output_file, input_file)
    
    print(f"기존 파일 백업: {backup_file}")
    print(f"✅ Ratio-321_fixed.jsonl 업데이트 완료! (총 {len(existing_data)}개)")

if __name__ == "__main__":
    add_one_to_ratio_321() 