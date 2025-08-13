#!/usr/bin/env python3
"""
데이터 타입과 클래스 비율을 고려한 균형잡힌 샘플을 생성하는 스크립트
"""

import json
import random
from collections import defaultdict

def analyze_data_distribution(data):
    """데이터 분포 분석"""
    type_class_counts = defaultdict(lambda: defaultdict(int))
    total_counts = defaultdict(int)
    
    for item in data:
        conversations = item.get('conversations', [])
        if len(conversations) >= 6:
            ground_truth = conversations[-1]['value']
            
            # 데이터 타입 추출 (이미지 경로에서)
            image_path = item.get('image', '')
            if 'crop' in image_path:
                data_type = 'crop'
            elif 'warp' in image_path:
                data_type = 'warp'
            elif 'flat' in image_path:
                data_type = 'flat'
            else:
                data_type = 'other'
            
            type_class_counts[data_type][ground_truth] += 1
            total_counts[ground_truth] += 1
    
    return type_class_counts, total_counts

def create_balanced_sample(data, sample_size=50):
    """균형잡힌 샘플 생성"""
    
    # 데이터 분포 분석
    type_class_counts, total_counts = analyze_data_distribution(data)
    
    print("=== 원본 데이터 분포 ===")
    print("클래스별 총 개수:")
    for class_name, count in total_counts.items():
        print(f"  {class_name}: {count}")
    
    print("\n데이터 타입별 클래스 분포:")
    for data_type, class_counts in type_class_counts.items():
        print(f"  {data_type}:")
        for class_name, count in class_counts.items():
            print(f"    {class_name}: {count}")
    
    # 샘플 크기 계산 (비율 유지)
    total_original = sum(total_counts.values())
    sample_ratios = {class_name: count/total_original for class_name, count in total_counts.items()}
    
    print(f"\n=== 샘플 크기 계산 ===")
    sample_sizes = {}
    for class_name, ratio in sample_ratios.items():
        sample_sizes[class_name] = max(1, int(sample_size * ratio))
        print(f"  {class_name}: {sample_sizes[class_name]} (비율: {ratio:.3f})")
    
    # 실제 샘플링
    selected_samples = []
    random.seed(42)  # 재현성
    
    for data_type, class_counts in type_class_counts.items():
        for class_name, count in class_counts.items():
            # 해당 타입과 클래스의 데이터 찾기
            candidates = []
            for item in data:
                conversations = item.get('conversations', [])
                if len(conversations) >= 6:
                    ground_truth = conversations[-1]['value']
                    image_path = item.get('image', '')
                    
                    # 데이터 타입 확인
                    item_type = 'other'
                    if 'crop' in image_path:
                        item_type = 'crop'
                    elif 'warp' in image_path:
                        item_type = 'warp'
                    elif 'flat' in image_path:
                        item_type = 'flat'
                    
                    if item_type == data_type and ground_truth == class_name:
                        candidates.append(item)
            
            # 샘플링
            target_size = min(sample_sizes[class_name], len(candidates))
            if target_size > 0:
                sampled = random.sample(candidates, target_size)
                selected_samples.extend(sampled)
                print(f"  {data_type}-{class_name}: {len(sampled)}개 선택")
    
    return selected_samples

def main():
    input_path = "/home/intern/banner_vis/data/experiments/validation_test/datasets/cot/test_abs.jsonl"
    output_path = "/home/intern/banner_vis/data/experiments/validation_test/datasets/cot/test_balanced_sample_50.jsonl"
    
    # 원본 데이터 로드
    print("원본 데이터 로드 중...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"원본 데이터 수: {len(data)}")
    
    # 균형잡힌 샘플 생성
    sample_data = create_balanced_sample(data, sample_size=50)
    
    print(f"\n=== 최종 샘플 ===")
    print(f"샘플 데이터 수: {len(sample_data)}")
    
    # 샘플 분포 확인
    sample_type_class_counts, sample_total_counts = analyze_data_distribution(sample_data)
    
    print("\n샘플 클래스 분포:")
    for class_name, count in sample_total_counts.items():
        print(f"  {class_name}: {count}")
    
    print("\n샘플 데이터 타입별 클래스 분포:")
    for data_type, class_counts in sample_type_class_counts.items():
        print(f"  {data_type}:")
        for class_name, count in class_counts.items():
            print(f"    {class_name}: {count}")
    
    # 샘플 데이터 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n샘플 데이터 저장 완료: {output_path}")

if __name__ == "__main__":
    main() 