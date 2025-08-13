#!/usr/bin/env python3
"""
현수막 OCR 전용 스크립트
현수막에서 텍스트만 추출하여 JSON 파일로 저장
"""

import os
import argparse
import json
import re
from datetime import datetime
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
    process_vision_info
)

# OCR 전용 프롬프트 (오타 수정 포함)
OCR_PROMPT = (
    "<obj>이 현수막</obj>"
    "위 현수막에 적힌 모든 텍스트를 정확히 추출해주세요. "
    "OCR 과정에서 발생할 수 있는 오타(숫자, 한글)를 자동으로 수정하고, "
    "추출된 텍스트만 출력하세요."
)

def clean_ocr_text(text: str) -> str:
    """OCR 결과를 정규화하고 정리"""
    if not text:
        return ""
    
    # 불필요한 설명 제거
    remove_patterns = [
        r"현수막에서\s*추출된\s*텍스트.*?:\s*",
        r"다음은\s*현수막에서\s*추출한\s*텍스트.*?:\s*",
        r"현수막에\s*적힌\s*텍스트.*?:\s*",
        r"출력\s*형식:\s*",
        r"\[추출된\s*텍스트만\s*출력\]",
    ]
    
    cleaned = text
    for pattern in remove_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    
    # 전화번호 형식 통일
    cleaned = re.sub(r'(\d{3})[)\s-]*(\d{3,4})[)\s-]*(\d{4})', r'\1-\2-\3', cleaned)
    
    # 날짜 형식 통일
    cleaned = re.sub(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', r'\1년 \2월 \3일', cleaned)
    
    # 공백 정리
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

def validate_ocr_text(text: str) -> bool:
    """OCR 결과가 유효한지 검증"""
    if not text or len(text.strip()) < 5:
        return False
    
    # 의미없는 텍스트 패턴
    meaningless_patterns = [
        r'^[^\w가-힣]*$',  # 한글/영문/숫자 없음
        r'이미지가.*어렵습니다',
        r'텍스트를.*추출할 수 없습니다',
        r'텍스트가.*보이지 않습니다'
    ]
    
    for pattern in meaningless_patterns:
        if re.search(pattern, text):
            return False
    
    return True

def run_ocr_inference(model, processor, device, image_path):
    """단일 이미지 OCR 수행"""
    img_name = os.path.basename(image_path)
    
    try:
        # 이미지 존재 확인
        if not os.path.exists(image_path):
            print(f"[ERROR] 이미지 파일이 존재하지 않음: {image_path}")
            return None
        
        # 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                    {"type": "text", "text": OCR_PROMPT},
                ],
            }
        ]
        
        # 비전 정보 처리
        image_inputs, video_inputs = process_vision_info(messages)
        
        if not image_inputs:
            print(f"[ERROR] 이미지 입력이 없음: {image_path}")
            return None
        
        # 메시지를 텍스트로 변환
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 모델 입력 준비
        model_inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        # 추론 실행
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,  # OCR은 더 긴 텍스트 필요
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
        
        # 생성된 토큰만 추출
        generated_ids_only = generated_ids[0][model_inputs["input_ids"].shape[1]:]
        ocr_text = processor.batch_decode([generated_ids_only], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # OCR 결과 정리
        ocr_text = clean_ocr_text(ocr_text)
        
        # OCR 결과 검증
        if not validate_ocr_text(ocr_text):
            print(f"[WARNING] OCR 결과가 유효하지 않음: {image_path}")
            return None
        
        return {
            "image_path": image_path,
            "image_name": img_name,
            "ocr_text": ocr_text,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"[ERROR] OCR 중 오류 발생 ({img_name}): {e}")
        return None

def run_ocr_for_checkpoint(checkpoint_name, images, max_images=None):
    """특정 체크포인트에 대한 OCR 실행"""
    print(f"\n=== {checkpoint_name} 체크포인트 OCR 시작 ===")
    
    # 체크포인트 설정 가져오기
    config = get_checkpoint_config(checkpoint_name)
    checkpoint_path = config["path"]
    gpu_id = config["gpu_id"]
    
    print(f"체크포인트 경로: {checkpoint_path}")
    print(f"GPU ID: {gpu_id}")
    
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
    
    # 이미지 수 제한
    if max_images and len(images) > max_images:
        images = images[:max_images]
        print(f"이미지 수를 {max_images}개로 제한했습니다.")
    
    # OCR 실행
    results = []
    print(f"총 {len(images)}개 이미지에 대해 OCR을 시작합니다...")
    
    try:
        for image_id, image_path in tqdm(images, desc=f"{checkpoint_name} OCR"):
            result = run_ocr_inference(model, processor, device, image_path)
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
    save_ocr_results(results, output_dir, checkpoint_name)
    
    print(f"=== {checkpoint_name} 체크포인트 OCR 완료 ===")
    return results

def save_ocr_results(results, output_dir, checkpoint_name):
    """OCR 결과를 파일로 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{checkpoint_name}_ocr_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 결과 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"OCR 결과가 저장되었습니다: {filepath}")
    
    # 요약 정보 저장
    summary_file = os.path.join(output_dir, f"{checkpoint_name}_ocr_summary.txt")
    successful_results = [r for r in results if r is not None]
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"체크포인트: {checkpoint_name}\n")
        f.write(f"작업: OCR 전용\n")
        f.write(f"총 이미지 수: {len(results)}\n")
        f.write(f"성공한 OCR 수: {len(successful_results)}\n")
        f.write(f"실패한 OCR 수: {len(results) - len(successful_results)}\n")
        f.write(f"실행 시간: {timestamp}\n\n")
        
        # OCR 품질 통계
        if successful_results:
            text_lengths = [len(r["ocr_text"]) for r in successful_results]
            avg_length = sum(text_lengths) / len(text_lengths)
            f.write(f"평균 텍스트 길이: {avg_length:.1f}자\n")
            f.write(f"최소 텍스트 길이: {min(text_lengths)}자\n")
            f.write(f"최대 텍스트 길이: {max(text_lengths)}자\n")
    
    return filepath

def run_all_checkpoints_ocr(data_file, max_images=None, checkpoint_names=None):
    """모든 체크포인트에 대한 OCR 실행"""
    print("=== 전체 체크포인트 OCR 시작 ===")
    print(f"데이터 파일: {data_file}")
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
    
    print(f"OCR할 체크포인트: {checkpoint_names}")
    
    # 각 체크포인트에 대해 OCR 실행
    all_results = {}
    for i, checkpoint_name in enumerate(checkpoint_names):
        try:
            print(f"\n{'='*50}")
            print(f"체크포인트 {i+1}/{len(checkpoint_names)}: {checkpoint_name}")
            print(f"{'='*50}")
            
            results = run_ocr_for_checkpoint(
                checkpoint_name, 
                images, 
                max_images
            )
            all_results[checkpoint_name] = results
            
            # 다음 체크포인트 전에 잠시 대기 (GPU 안정화)
            if i < len(checkpoint_names) - 1:
                print(f"다음 체크포인트 준비 중... (5초 대기)")
                import time
                time.sleep(5)
                
        except Exception as e:
            print(f"[ERROR] {checkpoint_name} 체크포인트 OCR 실패: {e}")
            continue
    
    print("=== 전체 체크포인트 OCR 완료 ===")
    return all_results

def test_typo_correction():
    """오타 수정 기능 테스트"""
    test_cases = [
        # 일반적인 OCR 오타
        "전화번호: 010-1234-5678",
        "전화번호: 010O12345678",  # O를 0으로 수정
        "전화번호: 010l2345678",   # l을 1로 수정
        
        # 한글 오타
        "안내사항이 있슴니다",
        "공지사항을 알내드립니다",
        "홍보물을 홍부합니다",
        
        # 날짜 형식
        "2024년1월15일",
        "2024년 1월 15일",
        
        # 주소 형식
        "서울시강남구역삼동",
        "서울시 강남구 역삼동",
        
        # 복합 오타
        "전화번호: 010O12345678, 안내사항이 있슴니다. 2024년1월15일까지",
    ]
    
    print("=== OCR 오타 수정 테스트 ===")
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n테스트 {i}:")
        print(f"원본: {test_text}")
        
        corrected_text, corrections = validate_and_fix_text(test_text)
        print(f"수정: {corrected_text}")
        
        if corrections:
            print("수정 사항:")
            for correction in corrections:
                print(f"  - {correction}")
        else:
            print("수정 사항 없음")
        
        # 품질 분석
        analysis = analyze_ocr_quality(corrected_text)
        print(f"품질 점수: {analysis['quality_score']:.1f}/100")
    
    print("\n=== 테스트 완료 ===")

def main():
    parser = argparse.ArgumentParser(description="현수막 OCR 전용 스크립트")
    parser.add_argument("--data_file", type=str, default=DATA_FILES["test_image_all"], 
                       help="OCR할 이미지가 포함된 JSONL 파일 경로")
    parser.add_argument("--max_images", type=int, default=None,
                       help="최대 이미지 수 (None이면 모든 이미지)")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                       help="OCR할 체크포인트 이름들 (None이면 모든 체크포인트)")
    parser.add_argument("--single_checkpoint", type=str, default=None,
                       help="단일 체크포인트만 OCR (checkpoints 옵션과 상호 배타적)")
    parser.add_argument("--test_typo", action="store_true",
                       help="오타 수정 기능 테스트 실행")
    
    args = parser.parse_args()
    
    # 오타 수정 테스트 모드
    if args.test_typo:
        test_typo_correction()
        return
    
    # 단일 체크포인트 모드
    if args.single_checkpoint:
        print(f"=== 단일 체크포인트 OCR 모드: {args.single_checkpoint} ===")
        
        # 이미지 로드
        images = load_images_from_jsonl(args.data_file, args.max_images)
        if not images:
            print("[ERROR] 로드할 이미지가 없습니다.")
            return
        
        # 단일 체크포인트 OCR
        results = run_ocr_for_checkpoint(
            args.single_checkpoint,
            images,
            args.max_images
        )
        
        if results:
            print(f"단일 체크포인트 OCR 완료: {args.single_checkpoint}")
        else:
            print(f"단일 체크포인트 OCR 실패: {args.single_checkpoint}")
    
    # 전체 체크포인트 모드
    else:
        run_all_checkpoints_ocr(
            args.data_file,
            args.max_images,
            args.checkpoints
        )

if __name__ == "__main__":
    main() 