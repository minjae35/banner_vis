#!/usr/bin/env python3
"""
간단한 현수막 분류 데이터셋 생성 스크립트
기존 CoT 데이터에서 이미지와 최종 분류 라벨만 추출하여 새로운 데이터셋 생성
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any

def extract_final_classification(conversations: List[Dict]) -> str:
    """대화에서 최종 분류 결과만 추출"""
    for conv in reversed(conversations):
        if conv["from"] == "gpt":
            value = conv["value"].strip()
            # 최종 분류 결과 찾기
            if value in ["정당", "공공", "민간"]:
                return value
            # "공공" 같은 형태로 끝나는 경우
            if value.endswith(("정당", "공공", "민간")):
                return value.split()[-1]
    return None

def create_simple_classification_data(input_file: str, output_file: str) -> None:
    """간단한 분류 데이터셋 생성"""
    simple_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # 최종 분류 결과 추출
            final_class = extract_final_classification(data["conversations"])
            if not final_class:
                print(f"Warning: No final classification found for {data.get('id', 'unknown')}")
                continue
            
            # 새로운 간단한 구조 생성 (이미지 경로는 그대로 유지)
            simple_item = {
                "id": data["id"],
                "image": data["image"],  # 원본 이미지 경로 그대로 유지
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\n이 현수막이 정당/공공/민간 중 어디에 해당하나요?"
                    },
                    {
                        "from": "gpt", 
                        "value": final_class
                    }
                ]
            }
            simple_data.append(simple_item)
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in simple_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created {len(simple_data)} simple classification samples")
    print(f"Output saved to: {output_file}")

def process_all_cot_files():
    """cot 폴더의 모든 파일을 처리"""
    cot_dir = "data/experiments/datasets/cot"
    output_dir = "data/experiments/datasets/simple_classification"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # cot 폴더의 모든 jsonl 파일 처리
    for filename in os.listdir(cot_dir):
        if filename.endswith('.jsonl'):
            input_path = os.path.join(cot_dir, filename)
            # 파일명에서 _fixed_abs 부분을 _simple_class로 변경
            output_filename = filename.replace('_fixed_abs.jsonl', '_simple_class.jsonl')
            output_path = os.path.join(output_dir, output_filename)
            
            print(f"Processing: {filename}")
            create_simple_classification_data(input_path, output_path)
            print(f"Completed: {output_filename}\n")

def main():
    parser = argparse.ArgumentParser(description="Create simple classification dataset")
    parser.add_argument("--input", help="Input CoT dataset file (optional)")
    parser.add_argument("--output", help="Output simple classification file (optional)")
    parser.add_argument("--process-all", action="store_true", help="Process all files in cot directory")
    
    args = parser.parse_args()
    
    if args.process_all:
        process_all_cot_files()
    elif args.input and args.output:
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} not found")
            return
        create_simple_classification_data(args.input, args.output)
    else:
        print("Usage:")
        print("  --process-all: Process all files in cot directory")
        print("  --input <file> --output <file>: Process single file")

if __name__ == "__main__":
    main() 