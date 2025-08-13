#!/usr/bin/env python3
"""
< 단일 체크포인트 추론 스크립트 >
Balanced Equal 체크포인트만 사용하여 src/sample_img의 이미지들에 대해 추론 수행
결과는 inference_results_single에 저장
"""

import os
import argparse
import glob
from tqdm import tqdm
import torch

from config import (
    get_checkpoint_config,
    RESULTS_BASE_DIR
)
from inference_utils import (
    load_model_and_processor,
    run_single_inference,
    save_results,
    get_prompt_text
)

def get_sample_images():
    """src/sample_img 디렉토리의 모든 이미지 파일 경로 반환"""
    sample_img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src/sample_img")
    
    # 지원하는 이미지 확장자
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    image_files = []
    
    # 메인 디렉토리의 이미지들
    for ext in image_extensions:
        pattern = os.path.join(sample_img_dir, ext)
        image_files.extend(glob.glob(pattern))
    
    # image 서브디렉토리의 이미지들
    image_subdir = os.path.join(sample_img_dir, "image")
    if os.path.exists(image_subdir):
        for ext in image_extensions:
            pattern = os.path.join(image_subdir, ext)
            image_files.extend(glob.glob(pattern))
    
    # 파일 경로 정렬
    image_files.sort()
    
    return image_files

def run_single_checkpoint_inference(checkpoint_name="bal_equal", prompt_type="standard"):      
    """Balanced Equal 체크포인트를 사용하여 sample_img의 이미지들에 대해 추론 실행"""
    print(f"\n=== {checkpoint_name} 체크포인트로 sample_img 추론 시작 ===")
    
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
    
    # sample_img의 이미지들 가져오기
    image_files = get_sample_images()
    if not image_files:
        print("[ERROR] src/sample_img에서 이미지를 찾을 수 없습니다.")
        return None
    
    print(f"발견된 이미지 수: {len(image_files)}")
    for img_path in image_files:
        print(f"  - {os.path.basename(img_path)}")
    
    # 모델과 프로세서 로드
    try:
        model, processor, device = load_model_and_processor(checkpoint_path, gpu_id)
        print("✅ 모델과 프로세서 로드 완료")
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        return None
    
    # 프롬프트 텍스트 가져오기
    prompt_text = get_prompt_text(prompt_type)
    print(f"✅ 프롬프트 텍스트:\n{prompt_text}")
    
    # 추론 실행
    results = []
    print(f"총 {len(image_files)}개 이미지에 대해 추론을 시작합니다...")
    
    try:
        for image_path in tqdm(image_files, desc=f"{checkpoint_name} 추론"):
            image_id = os.path.basename(image_path)
            result = run_single_inference(model, processor, device, image_path, prompt_text)
            
            if result:
                result["checkpoint_name"] = checkpoint_name
                result["image_id"] = image_id
                result["image_path"] = image_path
            results.append(result)
            
    finally:
        # 메모리 정리
        del model
        del processor
        torch.cuda.empty_cache()
        print(f"GPU 메모리 정리 완료 (GPU {gpu_id})")
    
    # 결과 저장
    output_dir = os.path.join("/home/intern2/banner_vis/experiments/inference/inference_results_single", checkpoint_name)       # output directory
    save_results(results, output_dir, checkpoint_name, prompt_type)
    
    print(f"=== {checkpoint_name} 체크포인트 추론 완료 ===")
    print(f"결과 저장 위치: {output_dir}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="단일 체크포인트 추론 스크립트 (Balanced Equal)")
    parser.add_argument("--checkpoint", type=str, default="bal_equal", 
                       help="사용할 체크포인트 이름 (기본값: bal_equal)")
    parser.add_argument("--prompt_type", type=str, default="standard", 
                       choices=["standard", "simple", "ocr_only"],
                       help="사용할 프롬프트 타입")
    
    args = parser.parse_args()
    
    print("=== 단일 체크포인트 추론 시작 ===")
    print(f"체크포인트: {args.checkpoint}")
    print(f"프롬프트 타입: {args.prompt_type}")
    print(f"이미지 소스: src/sample_img")
    print(f"결과 저장: inference_results_single")
    
    # 추론 실행
    results = run_single_checkpoint_inference(args.checkpoint, args.prompt_type)
    
    if results:
        print(f"\n✅ 추론 완료!")
        print(f"처리된 이미지 수: {len(results)}")
        print(f"결과 저장 위치: inference_results_single/{args.checkpoint}/")
    else:
        print(f"\n❌ 추론 실패!")

if __name__ == "__main__":
    main()


# ===================== # 
# python run_inference_single.py 
# python run_inference_single.py --checkpoint bal_equal --prompt_type standard
# python run_inference_single.py --ptompt_type simple
# ===================== #