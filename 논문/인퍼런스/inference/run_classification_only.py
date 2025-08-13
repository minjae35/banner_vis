#!/usr/bin/env python3
"""
현수막 분류 전용 스크립트
OCR 결과 파일을 읽어서 분류만 수행
"""

import os
import argparse
import json
import re
from datetime import datetime
from tqdm import tqdm
import torch
import glob

from config import (
    get_checkpoint_path, 
    get_checkpoint_gpu_id, 
    get_all_checkpoint_names,
    get_checkpoint_config,
    DATA_FILES,
    RESULTS_BASE_DIR
)
from inference_utils import (
    load_model_and_processor
)

# 분류 전용 프롬프트 템플릿
CLASSIFICATION_PROMPT_TEMPLATE = (
    "다음은 현수막에서 추출한 텍스트입니다:\n"
    "---\n"
    "{ocr_text}\n"
    "---\n\n"
    "위 현수막의 텍스트를 꼼꼼하게 읽고, 내용을 말해주세요.\n"
    "그 후 이 현수막이 '정당 현수막', '민간 현수막', '공공 현수막' 중 어디에 해당하는지 판단해주세요.\n\n"
    "## 분류 기준 (우선순위 순)\n\n"
    "### 1. 정당 현수막 (최우선)\n"
    "- 정당명, 후보자, 선거 관련 문구가 포함된 경우\n"
    "- 예시: \"민주당\", \"국민의힘\", \"○○ 후보\", \"선거\", \"투표\" 등\n\n"
    "### 2. 공공 현수막\n"
    "다음 중 하나라도 해당하는 경우:\n"
    "- **주최/주관이 정부/지자체/공공기관인 경우**\n"
    "  - 예시: \"○○시청\", \"○○구청\", \"○○공단\", \"○○부\", \"○○청\" 등이 주최/주관으로 명시\n"
    "- **정책, 단속, 행정 안내 등 공익적 내용**\n"
    "  - 예시: \"도로정비\", \"행정안전부\", \"공익사업\" 등\n"
    "- **공공기관의 공식 행사나 안내**\n"
    "  - 예시: \"○○공사 시무식\", \"○○청 안내\" 등\n\n"
    "### 3. 민간 현수막 (기본값)\n"
    "- 상업 광고, 학원/상점/개인 목적 문구\n"
    "- 민간 단체/개인의 홍보, 행사 안내\n"
    "- **단, 정부기관이 단순히 '후원'으로만 명시된 경우는 민간으로 분류**\n\n"
    "## 주의사항\n\n"
    "1. **주최/주관 vs 후원 구분**\n"
    "   - 주최/주관이 공공기관이면 → 공공 현수막\n"
    "   - 주최/주관이 민간이지만 공공기관이 후원이면 → 민간 현수막\n\n"
    "2. **복합적 상황 처리**\n"
    "   - 정당 관련 문구가 있으면 무조건 정당 현수막\n"
    "   - 그 다음으로 공공기관 주최/주관 여부 확인\n"
    "   - 나머지는 민간 현수막\n\n"
    "3. **명확하지 않은 경우**\n"
    "   - 공공적 성격이 뚜렷하면 공공 현수막\n"
    "   - 상업적/개인적 성격이 뚜렷하면 민간 현수막\n\n"
    "## 출력 형식\n\n"
    "- 현수막 내용: [현수막 내용 읽기]\n"
    "- 분류 결과: [정당 현수막 / 민간 현수막 / 공공 현수막]\n"
    "- 판단 이유: [분류 기준을 적용한 구체적인 근거 설명]\n"
    "  - 어떤 키워드나 문구가 해당 분류의 근거가 되는지 명시\n"
    "  - 주최/주관/후원 관계가 있다면 이를 명확히 구분하여 설명"
)

def parse_classification_result(text: str) -> dict:
    """분류 결과를 파싱"""
    result = {
        "classification": "민간",  # 기본값
        "reason": "",
        "content_summary": "",
        "raw_response": text
    }
    
    if not text:
        return result
    
    # 현수막 내용 요약 추출
    content_patterns = [
        r"현수막\s*내용:\s*(.+)",
        r"내용:\s*(.+)"
    ]
    
    for pattern in content_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            # 다음 섹션까지 추출
            if "분류 결과:" in content:
                content = content.split("분류 결과:")[0].strip()
            result["content_summary"] = content
            break
    
    # 분류 결과 추출
    classification_patterns = [
        r"분류\s*결과:\s*(정당|공공|민간)\s*현수막",
        r"분류\s*결과:\s*(정당|공공|민간)",
        r"(정당|공공|민간)\s*현수막",
        r"(정당|공공|민간)"
    ]
    
    for pattern in classification_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            result["classification"] = match.group(1)
            break
    
    # 판단 이유 추출
    reason_patterns = [
        r"판단\s*이유:\s*(.+)",
        r"근거:\s*(.+)",
        r"이유:\s*(.+)"
    ]
    
    for pattern in reason_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            reason = match.group(1).strip()
            # 다음 섹션이나 끝까지 추출
            if "\n\n" in reason:
                reason = reason.split("\n\n")[0].strip()
            result["reason"] = reason
            break
    
    return result

def run_classification_inference(model, processor, device, ocr_text):
    """단일 OCR 텍스트에 대한 분류 수행 (텍스트만 사용)"""
    try:
        if not ocr_text or not ocr_text.strip():
            print("[ERROR] OCR 텍스트가 없어 분류를 수행할 수 없습니다.")
            return None
        
        # 분류 프롬프트
        classification_prompt = CLASSIFICATION_PROMPT_TEMPLATE.format(ocr_text=ocr_text)
        
        # 텍스트 전용 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": classification_prompt},
                ],
            }
        ]
        
        # 메시지를 텍스트로 변환
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 모델 입력 준비 (텍스트만)
        model_inputs = processor(
            text=[text],
            images=[],  # 빈 이미지 리스트
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        # 추론 실행
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
        
        # 생성된 토큰만 추출
        generated_ids_only = generated_ids[0][model_inputs["input_ids"].shape[1]:]
        classification_result = processor.batch_decode([generated_ids_only], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return classification_result.strip()
        
    except Exception as e:
        print(f"[ERROR] 분류 중 오류 발생: {e}")
        return None

def load_ocr_results(ocr_file_path):
    """OCR 결과 파일 로드"""
    try:
        with open(ocr_file_path, 'r', encoding='utf-8') as f:
            ocr_results = json.load(f)
        
        # 유효한 OCR 결과만 필터링
        valid_results = []
        for result in ocr_results:
            if result and result.get("ocr_text") and result.get("image_path"):
                valid_results.append(result)
        
        print(f"OCR 결과 파일에서 {len(valid_results)}개의 유효한 결과를 로드했습니다.")
        return valid_results
        
    except Exception as e:
        print(f"[ERROR] OCR 결과 파일 로드 실패: {e}")
        return []

def find_latest_ocr_file(checkpoint_name):
    """가장 최근 OCR 결과 파일 찾기"""
    output_dir = os.path.join(RESULTS_BASE_DIR, checkpoint_name)
    if not os.path.exists(output_dir):
        print(f"[ERROR] 결과 디렉토리가 존재하지 않음: {output_dir}")
        return None
    
    # OCR 결과 파일 패턴
    pattern = os.path.join(output_dir, f"{checkpoint_name}_ocr_*.json")
    ocr_files = glob.glob(pattern)
    
    if not ocr_files:
        print(f"[ERROR] OCR 결과 파일을 찾을 수 없음: {pattern}")
        return None
    
    # 가장 최근 파일 선택
    latest_file = max(ocr_files, key=os.path.getctime)
    print(f"사용할 OCR 결과 파일: {latest_file}")
    return latest_file

def run_classification_for_checkpoint(checkpoint_name, ocr_file_path=None):
    """특정 체크포인트에 대한 분류 실행"""
    print(f"\n=== {checkpoint_name} 체크포인트 분류 시작 ===")
    
    # OCR 결과 파일 경로 결정
    if ocr_file_path is None:
        ocr_file_path = find_latest_ocr_file(checkpoint_name)
        if ocr_file_path is None:
            return None
    
    # OCR 결과 로드
    ocr_results = load_ocr_results(ocr_file_path)
    if not ocr_results:
        print("[ERROR] OCR 결과가 없습니다.")
        return None
    
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
    
    # 분류 실행
    results = []
    print(f"총 {len(ocr_results)}개 OCR 결과에 대해 분류를 시작합니다...")
    
    try:
        for ocr_result in tqdm(ocr_results, desc=f"{checkpoint_name} 분류"):
            # 분류 수행 (텍스트만 사용)
            classification_text = run_classification_inference(
                model, processor, device, ocr_result["ocr_text"]
            )
            
            if classification_text:
                # 분류 결과 파싱
                parsed_result = parse_classification_result(classification_text)
                
                # 결과 구성
                result = {
                    "image_path": ocr_result["image_path"],
                    "image_name": ocr_result["image_name"],
                    "ocr_text": ocr_result["ocr_text"],
                    "classification_text": classification_text,
                    "content_summary": parsed_result["content_summary"],
                    "classification": parsed_result["classification"],
                    "reason": parsed_result["reason"],
                    "timestamp": datetime.now().isoformat(),
                    "checkpoint_name": checkpoint_name,
                    "image_id": ocr_result.get("image_id", "")
                }
            else:
                result = None
            
            results.append(result)
    finally:
        # 메모리 정리
        del model
        del processor
        torch.cuda.empty_cache()
        print(f"GPU 메모리 정리 완료 (GPU {gpu_id})")
    
    # 결과 저장
    output_dir = os.path.join(RESULTS_BASE_DIR, checkpoint_name)
    save_classification_results(results, output_dir, checkpoint_name)
    
    print(f"=== {checkpoint_name} 체크포인트 분류 완료 ===")
    return results

def save_classification_results(results, output_dir, checkpoint_name):
    """분류 결과를 파일로 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{checkpoint_name}_classification_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 결과 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"분류 결과가 저장되었습니다: {filepath}")
    
    # 요약 정보 저장
    summary_file = os.path.join(output_dir, f"{checkpoint_name}_classification_summary.txt")
    successful_results = [r for r in results if r is not None]
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"체크포인트: {checkpoint_name}\n")
        f.write(f"작업: 분류 전용\n")
        f.write(f"총 OCR 결과 수: {len(results)}\n")
        f.write(f"성공한 분류 수: {len(successful_results)}\n")
        f.write(f"실패한 분류 수: {len(results) - len(successful_results)}\n")
        f.write(f"실행 시간: {timestamp}\n\n")
        
        # 분류 결과 통계
        if successful_results:
            classifications = [r["classification"] for r in successful_results]
            from collections import Counter
            class_counts = Counter(classifications)
            f.write("분류 결과 통계:\n")
            for class_name, count in class_counts.items():
                f.write(f"  {class_name}: {count}개\n")
    
    return filepath

def run_all_checkpoints_classification(checkpoint_names=None, ocr_file_paths=None):
    """모든 체크포인트에 대한 분류 실행"""
    print("=== 전체 체크포인트 분류 시작 ===")
    
    # 체크포인트 목록 결정
    if checkpoint_names is None:
        checkpoint_names = get_all_checkpoint_names()
    
    print(f"분류할 체크포인트: {checkpoint_names}")
    
    # 각 체크포인트에 대해 분류 실행
    all_results = {}
    for i, checkpoint_name in enumerate(checkpoint_names):
        try:
            print(f"\n{'='*50}")
            print(f"체크포인트 {i+1}/{len(checkpoint_names)}: {checkpoint_name}")
            print(f"{'='*50}")
            
            # OCR 파일 경로 결정
            ocr_file_path = None
            if ocr_file_paths and checkpoint_name in ocr_file_paths:
                ocr_file_path = ocr_file_paths[checkpoint_name]
            
            results = run_classification_for_checkpoint(checkpoint_name, ocr_file_path)
            all_results[checkpoint_name] = results
            
            # 다음 체크포인트 전에 잠시 대기 (GPU 안정화)
            if i < len(checkpoint_names) - 1:
                print(f"다음 체크포인트 준비 중... (5초 대기)")
                import time
                time.sleep(5)
                
        except Exception as e:
            print(f"[ERROR] {checkpoint_name} 체크포인트 분류 실패: {e}")
            continue
    
    print("=== 전체 체크포인트 분류 완료 ===")
    return all_results

def main():
    parser = argparse.ArgumentParser(description="현수막 분류 전용 스크립트")
    parser.add_argument("--ocr_file", type=str, default=None,
                       help="OCR 결과 JSON 파일 경로 (None이면 자동으로 최신 파일 찾기)")
    parser.add_argument("--checkpoints", nargs="+", default=None,
                       help="분류할 체크포인트 이름들 (None이면 모든 체크포인트)")
    parser.add_argument("--single_checkpoint", type=str, default=None,
                       help="단일 체크포인트만 분류 (checkpoints 옵션과 상호 배타적)")
    
    args = parser.parse_args()
    
    # 단일 체크포인트 모드
    if args.single_checkpoint:
        print(f"=== 단일 체크포인트 분류 모드: {args.single_checkpoint} ===")
        
        # 단일 체크포인트 분류
        results = run_classification_for_checkpoint(
            args.single_checkpoint,
            args.ocr_file
        )
        
        if results:
            print(f"단일 체크포인트 분류 완료: {args.single_checkpoint}")
        else:
            print(f"단일 체크포인트 분류 실패: {args.single_checkpoint}")
    
    # 전체 체크포인트 모드
    else:
        # OCR 파일 경로 매핑
        ocr_file_paths = None
        if args.ocr_file:
            # 모든 체크포인트에 대해 같은 OCR 파일 사용
            checkpoint_names = args.checkpoints or get_all_checkpoint_names()
            ocr_file_paths = {name: args.ocr_file for name in checkpoint_names}
        
        run_all_checkpoints_classification(
            args.checkpoints,
            ocr_file_paths
        )

if __name__ == "__main__":
    main() 