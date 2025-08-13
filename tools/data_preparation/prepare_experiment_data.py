#!/usr/bin/env python3
"""
실험용 데이터 준비 스크립트
분할된 인덱스를 실제 학습용 JSONL 파일로 변환합니다.
"""

import json
import os
from pathlib import Path
import argparse
from typing import Dict, List

def load_original_data(data_dir: str) -> Dict[str, Dict]:
    """원본 데이터를 로드합니다."""
    datasets = {
        'crop': f'{data_dir}/base_data/crop_7000/minimal_7000.jsonl',
        'flat': f'{data_dir}/base_data/flat_3500/flat_3500.jsonl', 
        'warp': f'{data_dir}/base_data/warp_4500/warp_4500.jsonl'
    }
    
    all_data = {}
    
    for dataset_type, jsonl_path in datasets.items():
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found")
            continue
            
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                index = f"{dataset_type}_{i}"
                all_data[index] = data
    
    return all_data

def create_experiment_jsonl(experiment_indices: List[str], all_data: Dict, output_path: str):
    """실험용 JSONL 파일을 생성합니다."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for index in experiment_indices:
            if index in all_data:
                data = all_data[index]
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
            else:
                print(f"Warning: Index {index} not found in original data")
    
    print(f"Created {output_file} with {len(experiment_indices)} samples")

def create_validation_test_jsonl(splits: Dict[str, List[str]], all_data: Dict, output_dir: str):
    """Validation과 Test JSONL 파일을 생성합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Validation 데이터 생성
    val_file = output_path / 'fixed_splits_validation.jsonl'
    with open(val_file, 'w', encoding='utf-8') as f:
        for index in splits['validation']:
            if index in all_data:
                data = all_data[index]
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Created validation file: {val_file} with {len(splits['validation'])} samples")
    
    # Test 데이터 생성
    test_file = output_path / 'fixed_splits_test.jsonl'
    with open(test_file, 'w', encoding='utf-8') as f:
        for index in splits['test']:
            if index in all_data:
                data = all_data[index]
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Created test file: {test_file} with {len(splits['test'])} samples")

def main():
    parser = argparse.ArgumentParser(description='Prepare experiment data for training')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory path')
    parser.add_argument('--splits_dir', type=str, default='data/splits', help='Splits directory path')
    parser.add_argument('--output_dir', type=str, default='data/splits', help='Output directory for JSONL files')
    
    args = parser.parse_args()
    
    print("Loading original data...")
    all_data = load_original_data(args.data_dir)
    print(f"Loaded {len(all_data)} samples from original data")
    
    # 고정 분할 로드
    splits_file = Path(args.splits_dir) / 'fixed_splits.json'
    with open(splits_file, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    
    # Validation과 Test JSONL 생성
    print("\nCreating validation and test JSONL files...")
    create_validation_test_jsonl(splits, all_data, args.output_dir)
    
    # 실험별 JSONL 생성
    experiments_dir = Path(args.splits_dir) / 'experiments'
    
    for exp_file in experiments_dir.glob('*.json'):
        exp_name = exp_file.stem
        
        with open(exp_file, 'r', encoding='utf-8') as f:
            experiment_indices = json.load(f)
        
        output_path = Path(args.output_dir) / f'{exp_name}.jsonl'
        print(f"\nCreating {exp_name} JSONL...")
        create_experiment_jsonl(experiment_indices, all_data, output_path)
    
    print("\nAll experiment data prepared successfully!")

if __name__ == "__main__":
    main() 