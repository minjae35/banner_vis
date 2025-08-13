#!/usr/bin/env python3
import os
import json
from PIL import Image
from datasets import load_dataset
from transformers import (
    TrainingArguments,
    Trainer,
    AutoProcessor,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model
import torch

# -----------------------------
# 0. 설정
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"  # 사용할 GPU 설정

# JSONL 저장 건너뛰기
SKIP_JSONL_SAVE = True
# 디버깅용 데이터 축소
DEBUG_SUBSET = True
DEBUG_SUBSET_SIZE = 10

# -----------------------------
# 1. 데이터셋 로딩 및 축소
# -----------------------------
dataset = load_dataset('json', data_files={'train': 'varco_ocr_finetune.jsonl'})['train']
print(f"▶️ 로드된 원본 데이터셋 크기: {len(dataset)}")
if DEBUG_SUBSET:
    dataset = dataset.select(range(min(DEBUG_SUBSET_SIZE, len(dataset))))
    print(f"▶️ DEBUG: 상위 {DEBUG_SUBSET_SIZE}개로 축소된 데이터셋 크기: {len(dataset)}")

# -----------------------------
# 2. 모델 로딩 & LoRA 래핑
# -----------------------------
model_name = "NCSOFT/VARCO-VISION-14B-HF"
from transformers import LlavaOnevisionForConditionalGeneration
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# -----------------------------
# 3. 전처리 함수 정의
# -----------------------------
def preprocess(example):
    img = Image.open(example['image'])
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)
    img = img.convert("RGB")

    convs = example['conversations']
    for conv in convs:
        if conv.get('role') == 'user':
            new_c, seen = [], False
            for c in conv.get('content', []):
                if c.get('type') == 'image' and not seen:
                    new_c.append(c); seen = True
                elif c.get('type') != 'image':
                    new_c.append(c)
            conv['content'] = new_c

    prompt = processor.apply_chat_template(convs, add_generation_prompt=True)
    print("▶ prompt 예시:", prompt[:80].replace("\n", " "), "…")
    print("▶ <image> 개수:", prompt.count('<image>'))

    inputs = processor(images=img, text=prompt, return_tensors="pt")
    pv = inputs['pixel_values']
    print("▶ pixel_values shape (pre-adjust):", pv.shape)
    if pv.dim() == 5:
        inputs['pixel_values'] = pv[0, 0]
    elif pv.dim() == 4 and pv.shape[0] == 1:
        inputs['pixel_values'] = pv[0]
    print("▶ pixel_values shape (post-adjust):", inputs['pixel_values'].shape)

    assistant_txt = ""
    for conv in convs:
        if conv.get('role') == 'assistant':
            for c in conv.get('content', []):
                if c.get('type') == 'text':
                    assistant_txt = c['text']
    labels = processor.tokenizer(assistant_txt, return_tensors="pt", padding=True).input_ids

    out = {k: (v.squeeze(0) if isinstance(v, torch.Tensor) and v.dim()>1 and v.shape[0]==1 else v)
           for k, v in inputs.items()}
    out['labels'] = labels.squeeze(0)
    return out

# -----------------------------
# 4. map 전처리
# -----------------------------
print("▶️ map 시작")
processed = dataset.map(
    preprocess,
    remove_columns=dataset.column_names,
    num_proc=1
)
print(f"▶️ map 완료: 샘플 수 {len(processed)}")

# -----------------------------
# 5. (선택) JSONL 저장
# -----------------------------
if not SKIP_JSONL_SAVE:
    print("▶️ JSONL 저장 시작…")
    with open("processed_varco_ocr_finetune.jsonl", "w", encoding="utf-8") as fout:
        for idx, ex in enumerate(processed, start=1):
            if idx % 1000 == 0:
                print(f"  • 저장된 샘플 {idx}/{len(processed)}")
            for k, v in ex.items():
                if isinstance(v, torch.Tensor):
                    ex[k] = v.tolist()
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("▶️ JSONL 저장 완료")

# -----------------------------
# 6. Trainer 세팅 및 DataCollator
# -----------------------------
# Seq2Seq용 DataCollator로 labels padding 지원
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=None
)

# -----------------------------
# 7. monkey-patch: 이미지 피처 기능 우회하여 split 오류 방지
# -----------------------------
from types import MethodType

def _dummy_get_image_features(self, pixel_values):
    # monkey-patch로 실제 split을 우회: 배치 전개 없이 그대로 반환
    batch_size = pixel_values.shape[0]
    # flatten patches as single sequence
    return pixel_values.view(batch_size, -1, pixel_values.shape[-1]) if pixel_values.dim() == 3 else pixel_values

# 바인딩
model.get_image_features = MethodType(_dummy_get_image_features, model)

# -----------------------------
# 8. Trainer 객체 생성
# -----------------------------
training_args = TrainingArguments(
    output_dir='./lora-varco-vision',
    per_device_train_batch_size=1,
    num_train_epochs=1,
    fp16=True,
    logging_steps=1,
    save_steps=500,
    report_to='none',
    max_steps=1
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed,
    data_collator=data_collator
)

# -----------------------------
# 9. DataLoader & 학습 테스트
# -----------------------------
print("▶️ Trainer 준비 완료, DataLoader 테스트…")
dl = trainer.get_train_dataloader()
batch = next(iter(dl))
print("▶️ 배치 shapes:", batch['pixel_values'].shape, batch['labels'].shape)

print("▶️ Trainer.train() 시작")
trainer.train()
print("▶️ 학습 완료")
