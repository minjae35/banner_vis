#!/usr/bin/env python3
"""
기존 분류 결과를 분석하는 스크립트
"""

import json
from collections import Counter

def analyze_results():
    # 결과 파일 로드
    with open("checkpoint_classification_results/classification_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    
    print(f"총 샘플 수: {len(results)}")
    
    # GT와 예측 분포
    gt_counter = Counter()
    pred_counter = Counter()
    correct = 0
    
    for result in results:
        gt = result['ground_truth']
        pred = result['predicted']
        
        gt_counter[gt] += 1
        pred_counter[pred] += 1
        
        # 정확도 계산 (정규화된 형태로)
        if normalize_label(gt) == normalize_label(pred):
            correct += 1
    
    print(f"\n=== Ground Truth 분포 ===")
    for label, count in gt_counter.most_common():
        print(f"{label}: {count}")
    
    print(f"\n=== 예측 분포 ===")
    for label, count in pred_counter.most_common():
        print(f"{label}: {count}")
    
    print(f"\n=== 정확도 ===")
    accuracy = correct / len(results)
    print(f"정확도: {accuracy:.3f} ({correct}/{len(results)})")
    
    # 클래스별 성능
    print(f"\n=== 클래스별 성능 ===")
    for class_name in ["정당", "공공", "민간"]:
        tp = fp = fn = 0
        
        for result in results:
            gt = normalize_label(result['ground_truth'])
            pred = normalize_label(result['predicted'])
            
            if gt == class_name and pred == class_name:
                tp += 1
            elif gt != class_name and pred == class_name:
                fp += 1
            elif gt == class_name and pred != class_name:
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (TP={tp}, FP={fp}, FN={fn})")
    
    # 잘못 분류된 예시들
    print(f"\n=== 잘못 분류된 예시들 (처음 10개) ===")
    wrong_count = 0
    for result in results:
        gt = normalize_label(result['ground_truth'])
        pred = normalize_label(result['predicted'])
        
        if gt != pred:
            print(f"ID: {result['image_id']}")
            print(f"GT: {result['ground_truth']} -> {gt}")
            print(f"Pred: {result['predicted']} -> {pred}")
            print(f"Response: {result['raw_response'][:100]}...")
            print("-" * 50)
            wrong_count += 1
            
            if wrong_count >= 10:
                break

def normalize_label(label):
    """레이블을 정규화 (짧은 형태로 변환)"""
    if label is None:
        return None
    
    # 긴 형태를 짧은 형태로 변환
    if "정당 현수막" in label:
        return "정당"
    elif "공공 현수막" in label:
        return "공공"
    elif "민간 현수막" in label:
        return "민간"
    
    # 이미 짧은 형태인 경우
    if label in ["정당", "공공", "민간"]:
        return label
    
    return None

if __name__ == "__main__":
    analyze_results() 