# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
from transformers import TrainerCallback

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# import qwenvl.train.trainer
# from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    print(f"[CONFIG] Vision: {model_args.tune_mm_vision}, MLP: {model_args.tune_mm_mlp}, LLM: {model_args.tune_mm_llm}")
    
    # vision encoder requires_grad 설정
    if model_args.tune_mm_vision:
        print("[INFO] Vision Encoder 활성화 중...")
        for p in model.visual.parameters():
            p.requires_grad = True
        visual_trainable = sum(1 for p in model.visual.parameters() if p.requires_grad)
        print(f"[INFO] Vision Encoder 파라미터: {visual_trainable}개 활성화")
    else:
        print("[INFO] Vision Encoder 비활성화")
        for p in model.visual.parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        print("[INFO] MLP 활성화")
        for p in model.visual.merger.parameters():
            p.requires_grad = True
    else:
        print("[INFO] MLP 비활성화")
        for p in model.visual.merger.parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        print("[INFO] LLM 활성화")
        for p in model.model.parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        print("[INFO] LLM 비활성화")
        for p in model.model.parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    # 전체 파라미터 요약 (DeepSpeed 초기화 전이므로 생략)
    # total = sum(p.numel() for p in model.parameters())
    # trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"[SUMMARY] 전체 파라미터: {total:,}, 학습 가능: {trainable:,}")
    # print("-" * 50)


def get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and 'loss' in logs:
            print(f"[LOG] step={state.global_step}, loss={logs['loss']:.4f}, grad_norm={logs.get('grad_norm', '-')}, lr={logs.get('learning_rate', '-')}, samples/s={logs.get('samples_per_second', '-')}, epoch={state.epoch}")
        # 일정 step마다 예측 샘플 출력 (100 step마다)
        if logs is not None and state.global_step % 100 == 0 and state.global_step > 0:
            trainer = kwargs.get('model', None)
            dataloader = kwargs.get('dataloader', None)
            if trainer is not None and dataloader is not None:
                try:
                    batch = next(iter(dataloader))
                    model = trainer
                    model.eval()
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=batch['input_ids'].to(model.device),
                            pixel_values=batch['pixel_values'].to(model.device),
                            max_new_tokens=32
                        )
                    # 디코딩
                    tokenizer = kwargs.get('tokenizer', None)
                    if tokenizer is not None:
                        pred = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                        label = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)[0]
                        print(f"[SAMPLE] step={state.global_step}\n  pred: {pred}\n  label: {label}")
                except Exception as e:
                    print(f"[SAMPLE] 예측 샘플 출력 실패: {e}")


class UnfreezeAfterDeepSpeed(TrainerCallback):
    def __init__(self):
        self.unfrozen = False
    
    def on_train_start(self, args, state, control, **kwargs):
        """DeepSpeed 엔진이 완전히 초기화된 시점에서 Vision Encoder 파라미터 재활성화"""
        self._unfreeze_vision_encoder(kwargs["model"], "on_train_start")
    
    def on_step_begin(self, args, state, control, **kwargs):
        """첫 번째 스텝에서도 파라미터 상태 확인"""
        if not self.unfrozen and state.global_step == 1:
            self._unfreeze_vision_encoder(kwargs["model"], "on_step_begin")
    
    def _unfreeze_vision_encoder(self, model, stage):
        """Vision Encoder 파라미터 재활성화 공통 로직"""
        print(f"[DEEPSPEED] Vision Encoder 파라미터 재활성화 ({stage})")
        
        # DeepSpeed 엔진에서 실제 모델 가져오기
        if hasattr(model, 'module'):
            actual_model = model.module
        else:
            actual_model = model
        
        # Vision Encoder 파라미터 재활성화
        for p in actual_model.visual.parameters():
            p.requires_grad = True
        
        # 실제 활성화된 파라미터 수 확인
        visual_trainable = sum(1 for p in actual_model.visual.parameters() if p.requires_grad)
        print(f"[SUCCESS] Vision Encoder 파라미터: {visual_trainable}개 활성화 완료")
        print(f"[SUCCESS] 이제 실제 학습이 가능한 상태입니다!")
        
        self.unfrozen = True


def train(attn_implementation="eager"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower():
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            trust_remote_code=True,
            low_cpu_mem_usage=False,  # 분산 환경 safetensors 버그 우회
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            min_pixels=451584,
            max_pixels=451584,
            resized_height=672,
            resized_width=672,
        ).image_processor
        data_args.model_type = "qwen2.5vl"
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            min_pixels=451584,
            max_pixels=451584,
            resized_height=672,
            resized_width=672,
        ).image_processor
        data_args.model_type = "qwenvl"

    # 모델 구조 및 파라미터 점검 코드(불필요한 프린트 제거)
    # print("==== Model named children ====")
    # for name, module in model.named_children():
    #     print(name, type(module))
    # print("=============================")
    # print("==== Model named parameters ====")
    # for n, p in model.named_parameters():
    #     print(n, p.shape)
    # print("=============================")
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=getattr(training_args, "model_max_length", 1024),
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    # set_model 이후 실제로 requires_grad=True인 파라미터 점검 (DeepSpeed 초기화 전이므로 생략)
    # trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    # print(f"[INFO] 학습 대상 파라미터 개수: {len(trainable)}")
    
    # DeepSpeed 초기화 전이므로 0개가 정상임을 명시
    # if len(trainable) == 0:
    #     print("[NOTE] DeepSpeed 초기화 전 - 파라미터가 임시로 0개로 표시됩니다")
    #     print("[NOTE] DeepSpeed 엔진 완성 후 실제 학습 가능한 파라미터가 활성화됩니다")

    if get_rank() == 0:
        pass
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # Trainer 생성 직전에 Vision Encoder 파라미터 강제 활성화
    if model_args.tune_mm_vision:
        print("[INFO] Vision Encoder 파라미터 최종 확인...")
        for p in model.visual.parameters():
            p.requires_grad = True
        visual_trainable = sum(1 for p in model.visual.parameters() if p.requires_grad)
        print(f"[INFO] Vision Encoder 파라미터: {visual_trainable}개")
    
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )
    
    # Trainer 초기화 후 최종 확인
    total_trainable = sum(1 for p in trainer.model.parameters() if p.requires_grad)
    print(f"[FINAL] 학습 가능한 파라미터: {total_trainable}개")
    print("=" * 50)
    
    # 커스텀 콜백 추가
    trainer.add_callback(PrintLossCallback())
    trainer.add_callback(UnfreezeAfterDeepSpeed()) # DeepSpeed 래핑 이후 Vision Encoder 파라미터 재활성화 콜백 추가
    trainer.train(resume_from_checkpoint=False)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="eager")
