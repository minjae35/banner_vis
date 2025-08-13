#!/usr/bin/env python3
"""
통합 추론 스크립트
모든 체크포인트에 대한 추론을 하나의 스크립트로 처리
"""

import os
import argparse
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import torch

from config import (
    get_checkpoint_path, 
    get_checkpoint_gpu_id, 
    get_all_checkpoint_names,
    get_checkpoint_config,
    DATA_FILES,
    RESULTS_BASE_DIR
)
from inference_utils import (
    load_model_and_processor,
    load_images_from_jsonl,
    run_single_inference,
    save_results,
    get_prompt_text
)

def run_inference_for_checkpoint(checkpoint_name, images, prompt_type="standard", max_images=None):
    """특정 체크포인트에 대한 추론 실행"""
    print(f"\n=== {checkpoint_name} 체크포인트 추론 시작 ===")
    
    # 체크포인트 설정 가져오기
    config = get_checkpoint_config(checkpoint_name)
    checkpoint_path = config["path"]
    gpu_id = config["gpu_id"]
    
    print(f"체크포인트 경로: {checkpoint_path}")
    print(f"GPU ID: {gpu_id}")
    print(f"프롬프트 타입: {prompt_type}")
    
    # 체크포인트 존재 확인
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] 체크포인트가 존재하지 않음: {checkpoint_path}")
        return None
    
    # 모델과 프로세서 로드
    try:
        model, processor, device = load_model_and_processor(checkpoint_path, gpu_id)
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        return None
    
    # 프롬프트 텍스트 가져오기
    prompt_text = get_prompt_text(prompt_type)
    
    # 이미지 수 제한
    if max_images and len(images) > max_images:
        images = images[:max_images]
        print(f"이미지 수를 {max_images}개로 제한했습니다.")
    
    # 추론 실행
    results = []
    print(f"총 {len(images)}개 이미지에 대해 추론을 시작합니다...")
    
    try:
        for image_id, image_path in tqdm(images, desc=f"{checkpoint_name} 추론"):
            result = run_single_inference(model, processor, device, image_path, prompt_text)
            if result:
                result["checkpoint_name"] = checkpoint_name
                result["image_id"] = image_id
            results.append(result)
    finally:
        # 메모리 정리
        del model
        del processor
        torch.cuda.empty_cache()
        print(f"GPU 메모리 정리 완료 (GPU {gpu_id})")
    
    # 결과 저장
    output_dir = os.path.join(RESULTS_BASE_DIR, checkpoint_name)
    save_results(results, output_dir, checkpoint_name, prompt_type)
    
    print(f"=== {checkpoint_name} 체크포인트 추론 완료 ===")
    return results

def run_all_checkpoints_inference(data_file, prompt_type="standard", max_images=None, checkpoint_names=None):
    """모든 체크포인트에 대한 추론 실행"""
    print("=== 전체 체크포인트 추론 시작 ===")
    print(f"데이터 파일: {data_file}")
    print(f"프롬프트 타입: {prompt_type}")
    print(f"최대 이미지 수: {max_images}")
    
    # 이미지 로드
    print("이미지 로딩 중...")
    images = load_images_from_jsonl(data_file, max_images)
    
    if not images:
        print("[ERROR] 로드할 이미지가 없습니다.")
        return
    
    # 체크포인트 목록 결정
    if checkpoint_names is None:
        checkpoint_names = get_all_checkpoint_names()
    
    print(f"추론할 체크포인트: {checkpoint_names}")
    
    # 각 체크포인트에 대해 추론 실행
    all_results = {}
    for i, checkpoint_name in enumerate(checkpoint_names):
        try:
            print(f"\n{'='*50}")
            print(f"체크포인트 {i+1}/{len(checkpoint_names)}: {checkpoint_name}")
            print(f"{'='*50}")
            
            results = run_inference_for_checkpoint(
                checkpoint_name, 
                images, 
                prompt_type, 
                max_images
            )
            all_results[checkpoint_name] = results
            
            # 다음 체크포인트 전에 잠시 대기 (GPU 안정화)
            if i < len(checkpoint_names) - 1:
                print(f"다음 체크포인트 준비 중... (5초 대기)")
                import time
                time.sleep(5)
                
        except Exception as e:
            print(f"[ERROR] {checkpoint_name} 체크포인트 추론 실패: {e}")
            continue
    
    print("=== 전체 체크포인트 추론 완료 ===")
    return all_results

def main():
    parser = argparse.ArgumentParser(description="통합 추론 스크립트")
    parser.add_argument("--data_file", type=str, default=DATA_FILES["test_image_all"], 
                       help="추론할 이미지가 포함된 JSONL 파일 경로")
    parser.add_argument("--prompt_type", type=str, default="standard", 
                       choices=["standard", "simple", "ocr_only"],
                       help="사용할 프롬프트 타입")
    parser.add_argument("--max_images", type=int, default=None,
                       help="최대 이미지 수 (None이면 모든 이미지)")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                       help="추론할 체크포인트 이름들 (None이면 모든 체크포인트)")
    parser.add_argument("--single_checkpoint", type=str, default=None,
                       help="단일 체크포인트만 추론 (checkpoints 옵션과 상호 배타적)")
    
    args = parser.parse_args()
    
    # 단일 체크포인트 모드
    if args.single_checkpoint:
        print(f"=== 단일 체크포인트 모드: {args.single_checkpoint} ===")
        
        # 이미지 로드
        images = load_images_from_jsonl(args.data_file, args.max_images)
        if not images:
            print("[ERROR] 로드할 이미지가 없습니다.")
            return
        
        # 단일 체크포인트 추론
        results = run_inference_for_checkpoint(
            args.single_checkpoint,
            images,
            args.prompt_type,
            args.max_images
        )
        
        if results:
            print(f"단일 체크포인트 추론 완료: {args.single_checkpoint}")
        else:
            print(f"단일 체크포인트 추론 실패: {args.single_checkpoint}")
    
    # 전체 체크포인트 모드
    else:
        run_all_checkpoints_inference(
            args.data_file,
            args.prompt_type,
            args.max_images,
            args.checkpoints
        )

if __name__ == "__main__":
    main() 