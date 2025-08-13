#!/usr/bin/env python3
import os
import torch
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoImageProcessor, TrainingArguments, Trainer
from transformers import LlavaOnevisionForConditionalGeneration
from peft import LoraConfig, get_peft_model

# 0. GPU 설정: CUDA 6번과 7번만 사용
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# 1. 모델 및 Processor 로드
model_name = "NCSOFT/VARCO-VISION-14B-HF"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)
processor = AutoProcessor.from_pretrained(model_name)
image_processor = AutoImageProcessor.from_pretrained(model_name)

# 2. LoRA 래핑
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 3. 스트리밍 모드로 데이터셋 로드 (.jsonl)
dataset = load_dataset(
    "json",
    data_files={"train": "varco_ocr_finetune.jsonl"},
    streaming=True
)["train"]

# 4. 전처리 함수: 이미지 → 단일 프레임, 텍스트 → 토크나이즈

def preprocess(example):
    # 이미지
    image = Image.open(example["image"]).convert("RGB")
    img_inputs = image_processor(image, return_tensors="pt")  # [1,3,384,384]
    pv = img_inputs.pixel_values[0].unsqueeze(0)               # [1,3,384,384]

    # 텍스트
    conversation = example["conversations"]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    txt_in = processor.tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)

    # 레이블
    assistant_text = ""
    for conv in conversation:
        if conv.get("role") == "assistant":
            for c in conv.get("content", []):
                if c.get("type") == "text":
                    assistant_text = c.get("text", "")
    lbl_in = processor.tokenizer(assistant_text, return_tensors="pt", padding=False, truncation=True)

    return {
        "pixel_values": pv.squeeze(0),           # [1,3,384,384]
        "input_ids":   txt_in.input_ids.squeeze(0),
        "labels":      lbl_in.input_ids.squeeze(0),
    }

# 5. with_transform으로 on-the-fly 전처리 연결
dataset = dataset.with_transform(preprocess)

# 6. 커스텀 collate: 배치 차원 합치기 및 패딩

def collate_fn(batch):
    pixel_vals = torch.stack([b["pixel_values"] for b in batch], dim=0)
    input_ids = [b["input_ids"] for b in batch]
    labels    = [b["labels"] for b in batch]

    inputs_padded = processor.tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt"
    )
    labels_padded = processor.tokenizer.pad(
        {"input_ids": labels}, padding=True, return_tensors="pt"
    )

    return {
        "pixel_values": pixel_vals,
        "input_ids":    inputs_padded.input_ids,
        "labels":       labels_padded.input_ids,
    }

# 7. 트레이닝 아규먼트 설정
training_args = TrainingArguments(
    output_dir='./lora-varco-vision',
    per_device_train_batch_size=2,
    num_train_epochs=1,
    fp16=True,
    save_steps=1000,
    logging_steps=100,
    report_to='none'
)

# 8. Trainer 생성 및 학습 실행
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)
trainer.train()
