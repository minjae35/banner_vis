import json
import os
from pathlib import Path

def convert_to_simple_structure(input_file, output_file):
    """
    기존 JSONL 파일을 CoT 없는 구조로 변환
    구조: 텍스트 추출 -> 바로 최종 답안
    """
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            # 기존 conversations에서 필요한 부분만 추출
            conversations = data['conversations']
            
            # 새로운 구조로 변환
            new_conversations = []
            
            # 첫 번째 질문 (텍스트 추출)
            if len(conversations) >= 2:
                new_conversations.append(conversations[0])  # 텍스트 추출 질문
                new_conversations.append(conversations[1])  # 텍스트 추출 답변
            
            # 마지막 질문과 답변 (최종 분류)
            if len(conversations) >= 6:
                new_conversations.append(conversations[-2])  # 최종 분류 질문
                new_conversations.append(conversations[-1])  # 최종 분류 답변
            
            # 새로운 데이터 구조
            new_data = {
                "id": data['id'],
                "image": data['image'],
                "conversations": new_conversations
            }
            
            converted_data.append(new_data)
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"변환 완료: {input_file} -> {output_file}")
    print(f"총 {len(converted_data)}개 데이터 변환됨")

def main():
    # 변환할 파일들
    datasets = [
        "BAL-Equal_fixed_abs.jsonl",
        "CW-Only_fixed_abs.jsonl", 
        "No-Warp_fixed_abs.jsonl",
        "C3F2W1_fixed_abs.jsonl"
    ]
    
    input_dir = "data/experiments"
    output_dir = "data/experiments/simple_cot"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset in datasets:
        input_file = os.path.join(input_dir, dataset)
        output_file = os.path.join(output_dir, dataset.replace("_fixed_abs", "_simple"))
        
        if os.path.exists(input_file):
            convert_to_simple_structure(input_file, output_file)
        else:
            print(f"파일을 찾을 수 없습니다: {input_file}")

if __name__ == "__main__":
    main() 