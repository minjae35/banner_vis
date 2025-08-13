import os
import json
from PIL import Image
import torch
from transformers import AutoTokenizer
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.mm_utils import tokenizer_image_token, process_images
from banner_visualized import load_det_model, preprocess_image, banDet
from datetime import datetime

# 설정
BASE_DIR = "/home/intern/banner_vis/dino_vlm_zeroshot_output"
IMAGE_DIR = "/home/intern/banner_vis/image"
CFG_PATH = 'det_config.json'
GPU = 7

# VLM 프롬프트 (기존 현수막 분석용)
prompt_text = (
    "<obj>이 현수막</obj>"
    "위 현수막의 텍스트를 꼼꼼하게 읽고, 내용을 말해주세요"
    "그 후 이 현수막이 '정당 현수막', '민간 현수막', '공공 현수막' 중 어디에 해당하는지도 함께 판단해주세요.\n\n"
    "판단 기준:\n"
    "1. 정당 현수막: 정당명, 후보자, 선거 관련 문구 포함\n"
    "2. 민간 현수막: 상업 광고, 학원/상점/개인 목적 문구 포함\n"
    "3. 공공 현수막: 정부, 지자체, 공공기관 명칭 또는 정책, 단속, 행정 안내 등 공익적 내용 포함\n\n"
    "출력 형식:\n"
    "- 현수막 내용 : [현수막 내용 읽기]\n"
    "- 분류 결과: [정당 현수막 / 민간 현수막 / 공공 현수막]\n"
    "- 판단 이유: [텍스트나 문맥을 근거로 분류 이유 설명]"
)

# VLM 모델 미리 로드
model_name = "NCSOFT/VARCO-VISION-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LlavaQwenForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map=f"cuda:{GPU}",
)
vision_tower = model.get_vision_tower()
image_processor = vision_tower.image_processor

# Detector 미리 로드
gpu_id = GPU
with open(CFG_PATH, 'r', encoding='utf-8') as f:
    config_data = json.load(f)
det_model_path = config_data["det"]["model_path"]
det_model, test_pipeline = load_det_model(gpu_id, det_model_path)

# 대상 폴더 리스트
folders = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
print(f"[INFO] 대상 폴더: {folders}")

for folder in folders:
    folder_path = os.path.join(BASE_DIR, folder)
    # 원본 이미지 경로 추정
    for ext in [".jpg", ".jpeg", ".png"]:
        image_path = os.path.join(IMAGE_DIR, folder + ext)
        if os.path.exists(image_path):
            break
    else:
        print(f"[WARNING] 원본 이미지 없음: {folder}")
        continue
    print(f"[INFO] Processing {image_path}")
    # Detector 전처리
    from easydict import EasyDict
    import banner_visualized
    params = EasyDict(banner_visualized.config.params)
    params.save_root_path = folder_path
    params.save_path = folder
    params.out_img_filename = f"{folder}_detected.jpg"
    imgs, ratio, original_size, imgs_array = preprocess_image(
        image_path, params, device=f"cuda:{GPU}" if torch.cuda.is_available() else "cpu"
    )
    # Detector 실행
    data = [{
        "video_name": image_path,
        "frame_id": i,
        "width": original_size[1],
        "height": original_size[0],
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "annotation": [],
        "banner_annotation": [],
    } for i in range(imgs.shape[0])]
    det_results = banDet(det_model, test_pipeline, imgs_array, imgs, params, data)
    width, height = original_size[1], original_size[0]
    image = Image.open(image_path).convert("RGB")
    results = []
    for idx, det in enumerate(det_results[0]):
        if len(det) < 5:
            continue
        x1, y1, x2, y2, conf = det[:5]
        norm_bbox = [x1 / width, y1 / height, x2 / width, y2 / height]
        cropped = image.crop((int(x1), int(y1), int(x2), int(y2)))
        crop_path = os.path.join(folder_path, f"varco_crop_{idx+1}.jpg")
        cropped.save(crop_path)
        # VLM 추론
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image"},
                ],
            }
        ]
        prompt = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer_image_token(prompt, tokenizer, -200, return_tensors="pt").unsqueeze(0).to(model.device)
        processed_images = process_images([cropped], image_processor, model.config)
        processed_images = [img_tensor.half().to(model.device) for img_tensor in processed_images]
        image_sizes = [cropped.size]
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                images=processed_images,
                image_sizes=image_sizes,
                do_sample=False,
                max_new_tokens=512,
                use_cache=True,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        try:
            parsed = json.loads(decoded)
        except Exception:
            parsed = None
        results.append({
            "bbox_normalized": norm_bbox,
            "bbox_abs": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(conf),
            "crop_path": crop_path,
            "vlm_raw": decoded,
            "vlm_json": parsed,
        })
    # out0_varco.json 저장
    annotation = []
    summary = {}
    for idx, r in enumerate(results):
        vlm_raw = r["vlm_raw"]
        ori_txt = ""
        ban_cls = ""
        reason = ""
        lines = vlm_raw.splitlines()
        for i, line in enumerate(lines):
            if "현수막 내용" in line:
                ori_txt = line.split(":", 1)[-1].strip()
                if i+1 < len(lines) and lines[i+1].strip():
                    if ori_txt:
                        ori_txt += " " + lines[i+1].strip()
                    else:
                        ori_txt = lines[i+1].strip()
            if "분류 결과" in line:
                ban_cls = line.split(":", 1)[-1].strip()
            if "판단 이유" in line:
                reason = line.split(":", 1)[-1].strip()
                if i+1 < len(lines) and lines[i+1].strip():
                    reason += " " + lines[i+1].strip()
        annotation.append({
            "bbox_normalized": [float(x) for x in r["bbox_normalized"]],
            "bbox_abs": [float(x) for x in r["bbox_abs"]],
            "confidence": float(r["confidence"]),
            "crop_path": r["crop_path"]
        })
        summary[str(idx)] = {
            "ori_txt": ori_txt,
            "ban_cls": ban_cls,
            "reason": reason
        }
    out_json = {
        "image_name": image_path,
        "width": width,
        "height": height,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "annotation": annotation,
        "banner_annotation": [],
        "summary": summary
    }
    out0_path = os.path.join(folder_path, "out0_varco.json")
    with open(out0_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"✅ Varco result saved to: {out0_path}") 