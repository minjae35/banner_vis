#!/usr/bin/env python3
"""
원본 Qwen2.5-VL-3B-Instruct 모델로 test_image 경로의 24장 이미지 추론
"""

import os
import json
import shutil
from datetime import datetime
from tqdm import tqdm
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from config import PROMPT_TEMPLATES, get_model_config, get_test_images

# config에서 프롬프트 가져오기
PROMPT_TEXT = PROMPT_TEMPLATES["standard"]

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

def safe_generate(model, inputs, max_new_tokens=256):
    return model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
    )

def run_one(img_path, processor, model, device):
    # message 구성 (file:// 로 주는 게 가장 안전)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(img_path)}"},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # vision inputs 추출
    image_inputs, video_inputs = process_vision_info(messages)
    
    # 비디오 입력이 없을 때는 videos 파라미터를 제외
    if video_inputs:
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    else:
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = safe_generate(model, inputs, max_new_tokens=256)
    
    # 생성된 토큰만 추출
    generated_ids = gen_ids[0][inputs["input_ids"].shape[1]:]
    decoded = processor.batch_decode([generated_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # 간단 파싱
    ori_txt = ban_cls = reason = ""
    for line in decoded.splitlines():
        if "현수막 내용" in line:  ori_txt = line.split(":",1)[-1].strip()
        if "분류 결과"   in line:  ban_cls = line.split(":",1)[-1].strip()
        if "판단 이유"   in line:  reason = line.split(":",1)[-1].strip()

    return decoded, ori_txt, ban_cls, reason

def main():
    # 설정 가져오기
    model_config = get_model_config("base")
    model_name = model_config["name"]
    model_path = model_config["path"]
    result_dir = "./original_model_inference_results"
    
    os.makedirs(result_dir, exist_ok=True)
    
    print("=== 원본 Qwen2.5-VL-3B-Instruct 모델 추론 시작 ===")
    print(f"모델: {model_name}")
    print(f"모델 경로: {model_path}")
    print(f"결과 저장 경로: {result_dir}")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # 모델 로드
    print("모델 로딩 중...")
    torch.backends.cuda.matmul.allow_tf32 = True
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else "auto",
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(model_path)
    
    # 테스트 이미지 가져오기
    test_images = get_test_images("all")
    
    print(f"테스트할 이미지들 (총 {len(test_images)}장):")
    for i, img_path in enumerate(test_images, 1):
        print(f"  {i}. {os.path.basename(img_path)}")
    
    all_results = []
    
        # 추론 실행
    print("\n추론 시작...")
    for img_path in tqdm(test_images, desc="추론 진행"):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(result_dir, img_name)
        os.makedirs(out_dir, exist_ok=True)
        
        try:
            # 이미지 로드 확인
            if not os.path.exists(img_path):
                print(f"[ERROR] 이미지 파일이 존재하지 않음: {img_path}")
                continue
            
            decoded, ori_txt, ban_cls, reason = run_one(img_path, processor, model, device)
            
            # 결과 저장
            rec = {
                "image_name": img_name,
                "image_path": img_path,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {"ori_txt": ori_txt, "ban_cls": ban_cls, "reason": reason},
                "vlm_raw": decoded,
            }
            
            # 개별 디렉토리에 저장
            with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            with open(os.path.join(out_dir, "output.txt"), "w", encoding="utf-8") as f:
                f.write(decoded)
            with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(PROMPT_TEXT)
            shutil.copy2(img_path, os.path.join(out_dir, os.path.basename(img_path)))
            
            all_results.append(rec)
            print(f"[SUCCESS] {img_name}: 추론 완료")
            
        except Exception as e:
            import traceback
            print(f"[ERROR] {img_name}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # 에러 정보 저장
            err = {"image_name": img_name, "image_path": img_path, "error": str(e)}
            with open(os.path.join(out_dir, "error.json"), "w", encoding="utf-8") as f:
                json.dump(err, f, ensure_ascii=False, indent=2)
            continue

    # 전체 결과 저장
    with open(os.path.join(result_dir, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 추론 완료!")
    print(f"결과 저장 위치: {result_dir}")
    print(f"처리된 이미지: {len(all_results)}개")

if __name__ == "__main__":
    main() 