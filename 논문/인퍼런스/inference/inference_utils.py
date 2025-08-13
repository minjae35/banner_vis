#!/usr/bin/env python3
"""
추론 공통 유틸리티 함수들
코드 중복을 줄이기 위한 공통 함수들
"""

import os
import json
import torch
from datetime import datetime
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from tqdm import tqdm
from config import PROMPT_TEMPLATES, DATA_FILES, get_model_path

def process_vision_info(messages):
    """Extract vision information from messages"""
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        for content in message["content"]:
            if content["type"] == "image":
                image_path = content["image"].replace("file://", "")
                try:
                    image = Image.open(image_path).convert('RGB')
                    image_inputs.append(image)
                except Exception as e:
                    print(f"이미지 로드 실패: {image_path}, 에러: {e}")
                    continue
    
    return image_inputs, video_inputs

def load_model_and_processor(checkpoint_path, device_id=0):
    """모델과 프로세서 로드"""
    pretrained_path = get_model_path("pretrained")
    print(f"프로세서 로딩 중... (PRETRAINED_PATH: {pretrained_path})")
    processor = AutoProcessor.from_pretrained(pretrained_path)
    
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    print(f"모델 로딩 중... (checkpoint: {checkpoint_path})")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else "auto",
        device_map={"": device_id},
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    
    return model, processor, device

def load_images_from_jsonl(jsonl_path, max_images=None):
    """JSONL 파일에서 이미지 경로들을 로드"""
    images = []
    
    if not os.path.exists(jsonl_path):
        print(f"JSONL 파일이 존재하지 않습니다: {jsonl_path}")
        return images
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if 'image' in data and data['image']:
                        image_path = data['image']
                        image_id = data.get('id', f'line_{line_num}')
                        
                        # 이미지 파일이 실제로 존재하는지 확인
                        if os.path.exists(image_path):
                            images.append((image_id, image_path))
                        else:
                            print(f"이미지 파일이 존재하지 않음: {image_path}")
                            
                        # 최대 이미지 수 제한
                        if max_images and len(images) >= max_images:
                            break
                            
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류 (라인 {line_num}): {e}")
                    continue
                    
    except Exception as e:
        print(f"JSONL 파일 읽기 오류: {e}")
    
    print(f"총 {len(images)}개의 이미지를 로드했습니다.")
    return images

def run_single_inference(model, processor, device, image_path, prompt_text):
    """단일 이미지 추론 실행"""
    img_name = os.path.basename(image_path)
    
    try:
        # 이미지 존재 확인
        if not os.path.exists(image_path):
            print(f"[ERROR] 이미지 파일이 존재하지 않음: {image_path}")
            return None
        
        # 프롬프트 텍스트 검증
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            print(f"[ERROR] 유효하지 않은 프롬프트 텍스트: {prompt_text}")
            return None
            
        # 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                    {"type": "text", "text": prompt_text.strip()},
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
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
        
        # 생성된 토큰만 추출
        generated_ids_only = generated_ids[0][model_inputs["input_ids"].shape[1]:]
        response_text = processor.batch_decode([generated_ids_only], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return {
            "image_path": image_path,
            "image_name": img_name,
            "prompt": prompt_text,
            "response": response_text,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"[ERROR] 추론 중 오류 발생 ({img_name}): {e}")
        return None

def save_results(results, output_dir, checkpoint_name, prompt_type="standard"):
    """결과를 파일로 저장"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{checkpoint_name}_{prompt_type}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # 결과 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"결과가 저장되었습니다: {filepath}")
    
    # 요약 정보 저장
    summary_file = os.path.join(output_dir, f"{checkpoint_name}_{prompt_type}_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"체크포인트: {checkpoint_name}\n")
        f.write(f"프롬프트 타입: {prompt_type}\n")
        f.write(f"총 이미지 수: {len(results)}\n")
        f.write(f"성공한 추론 수: {len([r for r in results if r is not None])}\n")
        f.write(f"실패한 추론 수: {len([r for r in results if r is None])}\n")
        f.write(f"실행 시간: {timestamp}\n")
    
    return filepath

def get_prompt_text(prompt_type="standard"):
    """프롬프트 텍스트 반환"""
    if prompt_type not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    return PROMPT_TEMPLATES[prompt_type] 