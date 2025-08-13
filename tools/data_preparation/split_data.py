#!/usr/bin/env python3
"""
데이터 분할 스크립트
README에 따라 고정 Hold-out (Val/Test) 분할을 생성합니다.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import os

def load_dataset_info(data_dir: str) -> pd.DataFrame:
    """데이터셋 정보를 로드합니다."""
    datasets = {
        'crop': f'{data_dir}/base_data/crop_7000/minimal_7000.jsonl',
        'flat': f'{data_dir}/base_data/flat_3500/flat_3500.jsonl', 
        'warp': f'{data_dir}/base_data/warp_4500/warp_4500.jsonl'
    }
    
    all_data = []
    
    for dataset_type, jsonl_path in datasets.items():
        if not os.path.exists(jsonl_path):
            print(f"Warning: {jsonl_path} not found")
            continue
            
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                # Banner ID 추출 (id 필드에서)
                banner_id = data.get('id', '')
                image_path = data.get('image', '')
                
                all_data.append({
                    'index': f"{dataset_type}_{i}",
                    'dataset_type': dataset_type,
                    'banner_id': banner_id,
                    'image_path': image_path,
                    'text': data.get('text', ''),
                    'bbox': data.get('bbox', [])
                })
    
    return pd.DataFrame(all_data)

def create_fixed_splits(df: pd.DataFrame, random_state: int = 42) -> Dict[str, List[str]]:
    """
    README에 따라 고정 Hold-out 분할을 생성합니다.
    
    Returns:
        Dict with keys: 'train', 'validation', 'test'
    """
    
    np.random.seed(random_state)
    
    # 각 타입별로 10%씩 Validation, 10%씩 Test로 분할
    splits = {'train': [], 'validation': [], 'test': []}
    
    for dataset_type in ['crop', 'flat', 'warp']:
        type_df = df[df['dataset_type'] == dataset_type]
        
        # Banner ID별로 그룹화하여 누수 방지
        banner_ids = type_df['banner_id'].unique()
        np.random.shuffle(banner_ids)
        
        # 10% Validation, 10% Test, 80% Train
        n_banners = len(banner_ids)
        n_val = max(1, int(n_banners * 0.1))
        n_test = max(1, int(n_banners * 0.1))
        
        val_banners = banner_ids[:n_val]
        test_banners = banner_ids[n_val:n_val + n_test]
        train_banners = banner_ids[n_val + n_test:]
        
        # 각 분할에 해당하는 데이터 추출
        val_data = type_df[type_df['banner_id'].isin(val_banners)]['index'].tolist()
        test_data = type_df[type_df['banner_id'].isin(test_banners)]['index'].tolist()
        train_data = type_df[type_df['banner_id'].isin(train_banners)]['index'].tolist()
        
        splits['validation'].extend(val_data)
        splits['test'].extend(test_data)
        splits['train'].extend(train_data)
    
    return splits

def create_experiment_datasets(splits: Dict[str, List[str]], df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    """
    실험별 데이터셋을 생성합니다.
    """
    
    train_df = df[df['index'].isin(splits['train'])]
    
    experiments = {
        'BAL-Equal': {
            'crop': 2800,
            'flat': 2800, 
            'warp': 2800
        },
        'CW-Only': {
            'crop': 2800,
            'flat': 0,
            'warp': 2800
        },
        'No-Warp': {
            'crop': 2800,
            'flat': 2800,
            'warp': 0
        }
    }
    
    experiment_datasets = {}
    
    for exp_name, composition in experiments.items():
        exp_data = []
        
        for dataset_type, count in composition.items():
            if count == 0:
                continue
                
            type_data = train_df[train_df['dataset_type'] == dataset_type]
            
            if len(type_data) < count:
                print(f"Warning: {dataset_type} has only {len(type_data)} samples, requested {count}")
                count = len(type_data)
            
            # 랜덤 샘플링
            selected_indices = np.random.choice(type_data.index, size=count, replace=False)
            exp_data.extend(type_data.loc[selected_indices]['index'].tolist())
        
        experiment_datasets[exp_name] = exp_data
    
    return experiment_datasets

def save_splits(splits: Dict[str, List[str]], output_dir: str):
    """분할 결과를 저장합니다."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 고정 분할 저장
    splits_file = output_path / 'fixed_splits.json'
    with open(splits_file, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)
    
    print(f"Fixed splits saved to {splits_file}")
    
    # 분할 통계 출력
    print("\n=== Split Statistics ===")
    for split_name, indices in splits.items():
        print(f"{split_name}: {len(indices)} samples")
    
    # 타입별 통계
    print("\n=== Type-wise Statistics ===")
    for split_name, indices in splits.items():
        type_counts = {}
        for idx in indices:
            dataset_type = idx.split('_')[0]
            type_counts[dataset_type] = type_counts.get(dataset_type, 0) + 1
        
        print(f"{split_name}:")
        for dataset_type, count in sorted(type_counts.items()):
            print(f"  {dataset_type}: {count}")

def save_experiment_datasets(experiment_datasets: Dict[str, Dict[str, List[str]]], output_dir: str):
    """실험별 데이터셋을 저장합니다."""
    output_path = Path(output_dir) / 'experiments'
    output_path.mkdir(parents=True, exist_ok=True)
    
    for exp_name, indices in experiment_datasets.items():
        exp_file = output_path / f'{exp_name}.json'
        with open(exp_file, 'w', encoding='utf-8') as f:
            json.dump(indices, f, indent=2, ensure_ascii=False)
        
        print(f"{exp_name} dataset saved to {exp_file}")
    
    # 실험별 통계 출력
    print("\n=== Experiment Statistics ===")
    for exp_name, indices in experiment_datasets.items():
        type_counts = {}
        for idx in indices:
            dataset_type = idx.split('_')[0]
            type_counts[dataset_type] = type_counts.get(dataset_type, 0) + 1
        
        print(f"{exp_name}:")
        for dataset_type, count in sorted(type_counts.items()):
            print(f"  {dataset_type}: {count}")
        print(f"  Total: {len(indices)}")

def main():
    parser = argparse.ArgumentParser(description='Create fixed data splits for banner classification experiments')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory path')
    parser.add_argument('--output_dir', type=str, default='data/splits', help='Output directory for splits')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    print("Loading dataset information...")
    df = load_dataset_info(args.data_dir)
    
    print(f"Total samples: {len(df)}")
    print(f"Type distribution: {df['dataset_type'].value_counts().to_dict()}")
    
    print("\nCreating fixed splits...")
    splits = create_fixed_splits(df, random_state=args.seed)
    
    print("\nCreating experiment datasets...")
    experiment_datasets = create_experiment_datasets(splits, df)
    
    print("\nSaving results...")
    save_splits(splits, args.output_dir)
    save_experiment_datasets(experiment_datasets, args.output_dir)
    
    print("\nData splitting completed successfully!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 