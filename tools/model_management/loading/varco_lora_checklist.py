import sys
import os
import torch
from PIL import Image

print("\n================ VARCO-VISION-14B-HF LoRA 체크리스트 ================\n")

# 1. 모델 구조 및 호환성 확인
try:
    from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
    print("[1-1] HuggingFace Transformers import: ✅")
except ImportError as e:
    print(f"[1-1] HuggingFace Transformers import: ❌ {e}")
    sys.exit(1)

try:
    from peft import LoraConfig, get_peft_model
    print("[1-2] PEFT(LoRA) import: ✅")
except ImportError as e:
    print(f"[1-2] PEFT(LoRA) import: ❌ {e}")
    print("[INFO] pip install peft==0.10.0 등 필요")
    sys.exit(1)

model_name = "NCSOFT/VARCO-VISION-14B-HF"

# 2. 모델 로딩 테스트
try:
    print(f"[2-1] 모델 로딩 시도: {model_name}")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",  # GPU 사용시 "auto"로 변경
        #attn_implementation="flash_attention_2"
    )
    print("[2-1] 모델 로딩: ✅")
    print(f"[2-2] 모델 클래스: {type(model)}")
    print(f"[2-3] 언어모델: {model.language_model.__class__.__name__ if hasattr(model, 'language_model') else 'N/A'}")
    print(f"[2-4] Vision Tower: {model.vision_tower.__class__.__name__ if hasattr(model, 'vision_tower') else 'N/A'}")
except Exception as e:
    print(f"[2-1] 모델 로딩: ❌ {e}")
    sys.exit(1)

try:
    processor = AutoProcessor.from_pretrained(model_name)
    print("[2-5] Processor 로딩: ✅")
except Exception as e:
    print(f"[2-5] Processor 로딩: ❌ {e}")
    sys.exit(1)

# 3. 샘플 추론 (텍스트만)
try:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image"},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # 샘플 이미지 (흑백 384x384)
    img = Image.new("RGB", (384, 384), color="white")
    inputs = processor(images=img, text=prompt, return_tensors='pt')
    if 'pixel_values' in inputs:
        print(f"[DEBUG] pixel_values dtype: {inputs['pixel_values'].dtype}")
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.float32)
        print(f"[DEBUG] pixel_values dtype after cast: {inputs['pixel_values'].dtype}")
    # input_ids는 long 타입 유지, pixel_values만 float16으로 변환
    for k in inputs:
        if torch.is_tensor(inputs[k]):
            inputs[k] = inputs[k].to(model.device)
    print("[3-1] 샘플 입력 생성: ✅")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    print("[3-2] 샘플 추론: ✅")
except Exception as e:
    print(f"[3-2] 샘플 추론: ❌ {e}")
    sys.exit(1)

# 4. LoRA 래핑 시도
try:
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    lora_model = get_peft_model(model, lora_config)
    print("[4-1] PEFT LoRA 래핑: ✅")
    print(f"[4-2] 래핑 후 모델 클래스: {type(lora_model)}")
except Exception as e:
    print(f"[4-1] PEFT LoRA 래핑: ❌ {e}")
    print("[INFO] target_modules는 실제 구조에 따라 조정 필요. (예: q_proj, v_proj, o_proj 등)")
    sys.exit(1)

print("\n================ 모든 체크 완료! 이 모델은 LoRA 파인튜닝이 가능합니다. ================\n") 