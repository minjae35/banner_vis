#!/usr/bin/env python3
import json
import os

def fix_image_paths(input_file, output_file):
    """JSONL 파일의 이미지 경로를 절대 경로로 수정"""
    fixed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # 이미지 경로를 절대 경로로 수정
            if data['image'].startswith('data/base_data/'):
                data['image'] = '/home/intern/banner_vis/' + data['image']
            
            fixed_data.append(data)
    
    # 수정된 데이터를 새 파일에 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in fixed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Fixed {len(fixed_data)} items in {output_file}")

# 실험 데이터셋들 수정
experiment_files = [
    'data/experiments/BAL-Equal_fixed.jsonl',
    'data/experiments/CW-Only_fixed.jsonl', 
    'data/experiments/No-Warp_fixed.jsonl',
    'data/experiments/C3F2W1_fixed.jsonl'
]

# Validation/Test 데이터셋들 수정
validation_test_files = [
    'data/validation_test/validation.jsonl',
    'data/validation_test/test.jsonl'
]

print("Fixing experiment datasets...")
for file_path in experiment_files:
    output_path = file_path.replace('.jsonl', '_abs.jsonl')
    fix_image_paths(file_path, output_path)

print("\nFixing validation/test datasets...")
for file_path in validation_test_files:
    output_path = file_path.replace('.jsonl', '_abs.jsonl')
    fix_image_paths(file_path, output_path)

print("\nAll files have been fixed with absolute paths!") 