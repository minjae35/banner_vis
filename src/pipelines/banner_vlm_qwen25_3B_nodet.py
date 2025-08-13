#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, shutil
import sys
from datetime import datetime
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# qwen_vl_utils 경로 추가
sys.path.append('../qwen-tuning/Qwen2.5-VL/qwen-vl-utils/src')
from qwen_vl_utils import process_vision_info   # 핵심!
# -------------------------------------------------
# 설정
BASE_NAMES = [
    "20240904_091419",
    "20240907_152822",
    "20240909_095252_001",
    "20240909_173202",
    "20240911_082435",
    "20240914_085149_001",
    "A01_240929_HD_IMG_4428",
    "C04_240929_f3_20210916_134526",
    "IMG_3651",
    "O08_240929_bM_20230316_093209",
]
EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
RESULT_ROOT = "output_nodet_original"           
PROMPT_TEXT = (
    "<obj>이 현수막</obj>위 현수막의 텍스트를 꼼꼼하게 읽고, 내용을 말해주세요"
    "위 현수막의 텍스트를 꼼꼼하게 읽고, 내용을 말해주세요"
    "그 후 이 현수막이 '정당 현수막', '민간 현수막', '공공 현수막' 중 어디에 해당하는지도 함께 판단해주세요.\n\n"
    "판단 기준:\n"
    "1. 정당 현수막: 정당명, 후보자, 선거 관련 문구 포함\n"
    "2. 민간 현수막: 상업 광고, 학원/상점/개인 목적 문구 포함\n"
    "3. 공공 현수막: 정부, 지자체, 공공기관 명칭 또는 정책, 단속, 행정 안내 등 공익적 내용 포함\n\n"
    "출력 형식:\n"
    "- 현수막 내용 : 현수막 내용 읽기\n"
    "- 분류 결과: 정당 현수막 / 민간 현수막 / 공공 현수막\n"
    "- 판단 이유: 텍스트나 문맥을 근거로 분류 이유 설명"
)
# -------------------------------------------------

def args_parse():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--image_dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    # args.image_dir = '/home/intern2/banner_vis/src/pipelines'           # input image directory
    args.image_dir = '/home/intern2/banner_vis/src/sample_img/image'
    return args

def find_path(root, name):
    for e in EXTS:
        p = os.path.join(root, f"{name}{e}")
        if os.path.exists(p): return p
    # fallback: recursive
    for e in EXTS:
        for p in [os.path.join(root, "**", f"{name}{e}")]:
            hits = [h for h in glob.glob(p, recursive=True)]
            if hits: return hits[0]
    return None

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
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        gen_ids = safe_generate(model, inputs, max_new_tokens=256)
    trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], gen_ids)]
    decoded = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]      # decode (token id -> text)
    print(f"✅ decoded:\n {decoded}")

    # 간단 파싱
    ori_txt = ban_cls = reason = ""
    for line in decoded.splitlines():
        if "현수막 내용" in line:  ori_txt = line.split(":",1)[-1].strip()      # 내용만 추출
        if "분류 결과"   in line:  ban_cls = line.split(":",1)[-1].strip()     # 내용만 추출
        if "판단 이유"   in line:  reason = line.split(":",1)[-1].strip()      # 내용만 추출

    # 줄바꿈을 공백으로 치환하여 한 줄로 저장
    decoded_single_line = decoded.replace('\n', ' ').replace('\r', ' ')

    return decoded_single_line, ori_txt, ban_cls, reason          # ori_text: 현수막 내용(VLM이 읽은 내용), ban_cls: 분류 결과, reason: 판단 이유

def main():
    args = args_parse()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # CUDA_VISIBLE_DEVICES 사용 시 cuda:0
    os.makedirs(RESULT_ROOT, exist_ok=True)               # output directory

    # 모델 로드
    torch.backends.cuda.matmul.allow_tf32 = True
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else "auto",
        low_cpu_mem_usage=True,
    ).to(device).eval()

    # 해상도 범위 조절 필요하면 min/max_pixels 설정
    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        # min_pixels=256*28*28,
        # max_pixels=1280*28*28,
    )

    # 이미지 파일 검색 및 수집
    # 이미지 리스트: 폴더 내 모든 이미지 (서브디렉토리 포함)
    import glob
    image_files = []
    for ext in EXTS:               
        # 메인 디렉토리와 서브디렉토리에서 이미지 검색
        image_files.extend(glob.glob(os.path.join(args.image_dir, f"*{ext}")))
        image_files.extend(glob.glob(os.path.join(args.image_dir, "*", f"*{ext}")))
    image_files = sorted(image_files)
    print(f"[INFO] 총 {len(image_files)}개 이미지 추론: {image_files}")

    all_results = []
    for img_path in tqdm(image_files):
        name = os.path.splitext(os.path.basename(img_path))[0]
        out_dir = os.path.join(RESULT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)
        # VLM 분석 실행
        try:
            decoded, ori_txt, ban_cls, reason = run_one(img_path, processor, model, device)
            rec = {
                "image_name": name,
                "image_path": img_path,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {"ori_txt": ori_txt, "ban_cls": ban_cls, "reason": reason},              # ori_text: 현수막 내용(VLM이 읽은 내용), ban_cls: 분류 결과, reason: 판단 이유
                "vlm_raw": decoded,
            }
            # 결과 파일 저장
            with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            with open(os.path.join(out_dir, "output.txt"), "w", encoding="utf-8") as f:
                f.write(decoded)
            with open(os.path.join(out_dir, "prompt.txt"), "w", encoding="utf-8") as f:
                f.write(PROMPT_TEXT)
            shutil.copy2(img_path, os.path.join(out_dir, os.path.basename(img_path)))
            all_results.append(rec)
        # 오류 처리
        except Exception as e:
            err = {"image_name": name, "image_path": img_path, "error": str(e)}
            with open(os.path.join(out_dir, "error.json"), "w", encoding="utf-8") as f:
                json.dump(err, f, ensure_ascii=False, indent=2)
            print(f"[ERROR] {name}: {e}")

    # 전체 결과 통합 저장
    with open(os.path.join(RESULT_ROOT, "vlm_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("✅ Done.")

if __name__ == "__main__":
    import glob
    main()
