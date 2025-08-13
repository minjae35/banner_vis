#!/usr/bin/env python3
import json
import re

def clean_jsonl():
    # 파일 경로
    input_file = "/home/intern/banner_vis/data/merged_9500/warp/warp_4500.jsonl"
    output_file = "/home/intern/banner_vis/data/merged_9500/warp/warp_4500_cleaned.jsonl"
    
    # 제거할 패턴들
    remove_patterns = [
        r"현수막에 적힌 텍스트는 다음과 같습니다[:\s]*",
        r"현수막에서 추출한 텍스트는 다음과 같습니다[:\s]*",
        r"현수막에서 보이는 텍스트는 다음과 같습니다[:\s]*",
        r"현수막에 적힌 내용은 다음과 같습니다[:\s]*",
        r"현수막에서 확인할 수 있는 텍스트는 다음과 같습니다[:\s]*"
    ]
    
    # 삭제할 샘플의 키워드들
    delete_keywords = [
        "이미지가 흐릿하여 텍스트를 정확히 추출하기 어렵습니다",
        "이미지에서 텍스트를 추출할 수 없습니다",
        "텍스트를 정확히 추출하기 어렵습니다",
        "텍스트를 추출할 수 없습니다",
        "이미지가 흐릿합니다",
        "텍스트가 보이지 않습니다"
    ]
    
    print("JSONL 파일 읽는 중...")
    cleaned_samples = []
    deleted_count = 0
    cleaned_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                sample = json.loads(line.strip())
                
                # conversations에서 OCR 텍스트 찾기 (두 번째 gpt 응답)
                conversations = sample.get('conversations', [])
                if len(conversations) >= 2:
                    ocr_text = conversations[1].get('value', '')
                    
                    # 삭제할 키워드가 있는지 확인
                    should_delete = False
                    for keyword in delete_keywords:
                        if keyword in ocr_text:
                            should_delete = True
                            break
                    
                    if should_delete:
                        deleted_count += 1
                        continue
                    
                    # 텍스트 정리
                    cleaned_text = ocr_text
                    for pattern in remove_patterns:
                        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
                    
                    # 앞뒤 공백 제거
                    cleaned_text = cleaned_text.strip()
                    
                    # 텍스트가 변경되었으면 업데이트
                    if cleaned_text != ocr_text:
                        conversations[1]['value'] = cleaned_text
                        cleaned_count += 1
                
                cleaned_samples.append(sample)
                
            except json.JSONDecodeError as e:
                print(f"JSON 파싱 오류 (라인 {line_num}): {e}")
                continue
    
    print(f"총 샘플 수: {len(cleaned_samples)}")
    print(f"삭제된 샘플 수: {deleted_count}")
    print(f"정리된 샘플 수: {cleaned_count}")
    
    # 정리된 파일 저장
    print("정리된 JSONL 파일 저장 중...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in cleaned_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n=== 완료 ===")
    print(f"원본 파일: {input_file}")
    print(f"정리된 파일: {output_file}")
    print(f"정리된 샘플 수: {len(cleaned_samples)}")
    print(f"삭제된 샘플 수: {deleted_count}")
    print(f"텍스트 정리된 샘플 수: {cleaned_count}")

if __name__ == "__main__":
    clean_jsonl() 