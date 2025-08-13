#!/usr/bin/env python3
"""
테스트 데이터에서 샘플을 추출하는 스크립트
"""

import json
import random

def create_sample_data(input_path, output_path, sample_size=50):
    """테스트 데이터에서 샘플을 추출"""
    
    # 원본 데이터 로드
    with open(input_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"원본 데이터 수: {len(data)}")
    
    # 랜덤 샘플링
    random.seed(42)  # 재현성을 위한 시드 설정
    sample_data = random.sample(data, min(sample_size, len(data)))
    
    print(f"샘플 데이터 수: {len(sample_data)}")
    
    # 샘플 데이터 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"샘플 데이터 저장 완료: {output_path}")
    
    # 클래스 분포 확인
    class_counts = {}
    for item in sample_data:
        conversations = item.get('conversations', [])
        if len(conversations) >= 6:
            ground_truth = conversations[-1]['value']
            class_counts[ground_truth] = class_counts.get(ground_truth, 0) + 1
    
    print(f"\n샘플 클래스 분포:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

if __name__ == "__main__":
    input_path = "/home/intern/banner_vis/data/experiments/validation_test/datasets/cot/test_abs.jsonl"
    output_path = "/home/intern/banner_vis/data/experiments/validation_test/datasets/cot/test_sample_50.jsonl"
    
    create_sample_data(input_path, output_path, sample_size=50) 