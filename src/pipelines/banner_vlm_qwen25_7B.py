import os
import sys
import json
import time
import argparse
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# Detector imports (from banner_visualized.py)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from banner_visualized import load_det_model, preprocess_image, banDet

def crop_bbox(image, bbox):
    width, height = image.size
    x1 = int(bbox[0] * width)
    y1 = int(bbox[1] * height)
    x2 = int(bbox[2] * width)
    y2 = int(bbox[3] * height)
    return image.crop((x1, y1, x2, y2))

def get_args():
    parser = argparse.ArgumentParser(description='Qwen2.5-VL Detector-Cropped Inference')
    # parser.add_argument('--image', required=True, help='Input image path')
    args = parser.parse_args()
    args.image = '/home/intern/banner_vis/src/pipelines/test.jpg'
    args.cfg = '/home/intern/banner_vis/configs/det_config.json'
    args.out = 'output'
    args.gpu = 7
    return args

def main():
    args = get_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    img_basename = os.path.splitext(os.path.basename(args.image))[0]
    print("==================== Qwen2.5-VL DETECTOR INTEGRATED PIPELINE ====================")
    print(f"Image: {args.image}")
    print(f"Config: {args.cfg}")
    print(f"Output dir: {args.out}")
    print(f"Device: {device}\n")

    # 1. Load image
    image = Image.open(args.image).convert("RGB")
    width, height = image.size
    print(f"[INFO] Image loaded: {image.size} (W x H)")

    # 2. Load detector model
    with open(args.cfg, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    det_model_path = config_data["det"]["model_path"]
    gpu_id = args.gpu
    print("[INFO] Loading detector model...")
    det_model, test_pipeline = load_det_model(gpu_id, det_model_path)
    print("[INFO] Detector model loaded.")

    # 3. Preprocess image for detector
    from easydict import EasyDict
    import banner_visualized
    params = EasyDict(banner_visualized.config.params)
    params.save_root_path = args.out
    params.save_path = img_basename
    params.out_img_filename = f"{img_basename}_detected.jpg"
    imgs, ratio, original_size, imgs_array = preprocess_image(
        args.image, params, device=device
    )
    print(f"[INFO] Preprocessed image for detection. imgs.shape: {imgs.shape}")

    # 4. Run detection
    data = [{
        "video_name": args.image,
        "frame_id": i,
        "width": original_size[1],
        "height": original_size[0],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "annotation": [],
        "banner_annotation": [],
    } for i in range(imgs.shape[0])]
    print("[INFO] Running detection...")
    det_results = banDet(det_model, test_pipeline, imgs_array, imgs, params, data)
    print(f"[INFO] Detection results shape: {det_results.shape}")
    if det_results.numel() == 0 or det_results.shape[1] == 0:
        print("[WARNING] No banners detected! Exiting.")
        return
    print(f"[INFO] {det_results.shape[1]} banner(s) detected.")

    # 5. Load Qwen2.5-VL model & processor
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    print("[INFO] Loading Qwen2.5-VL model...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    print("[INFO] Qwen2.5-VL model loaded.")

    # 6. For each bbox, crop and run VLM
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
    results = []
    for idx, det in enumerate(det_results[0]):
        if len(det) < 5:
            continue
        x1, y1, x2, y2, conf = det[:5]
        scale_x = width / 1024
        scale_y = height / 1024
        x1_orig = float(x1) * scale_x
        y1_orig = float(y1) * scale_y
        x2_orig = float(x2) * scale_x
        y2_orig = float(y2) * scale_y
        norm_bbox = [x1_orig / width, y1_orig / height, x2_orig / width, y2_orig / height]
        cropped = crop_bbox(image, norm_bbox)
        crop_path = os.path.join(args.out, f"{img_basename}_crop_{idx+1}.jpg")
        cropped.save(crop_path)
        print(f"[INFO] Cropped banner saved: {crop_path} (size: {cropped.size})")

        # Prepare message for Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": cropped},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[cropped],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Run inference
        print(f"[INFO] Running Qwen2.5-VL inference for banner {idx+1}...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            decoded = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
        print(f"[INFO] Qwen2.5-VL raw output for banner {idx+1}: {decoded}")

        results.append({
            "bbox_normalized": [float(x) for x in norm_bbox],
            "bbox_abs": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(conf),
            "crop_path": crop_path,
            "vlm_raw": decoded,
        })

    # 결과 저장 등은 기존과 동일하게 추가 가능
    # out0.json: 전체 메타+annotation+summary 구조로 저장 (banner_r5/out0.json 최대한 유사)
    from datetime import datetime
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
            "bbox_normalized": r["bbox_normalized"],
            "bbox_abs": r["bbox_abs"],
            "confidence": r["confidence"],
            "crop_path": r["crop_path"]
        })
        summary[str(idx)] = {
            "ori_txt": ori_txt,
            "ban_cls": ban_cls,
            "reason": reason
        }
    out_json = {
        "image_name": args.image,
        "width": width,
        "height": height,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "annotation": annotation,
        "banner_annotation": [],
        "summary": summary
    }
    out0_path = os.path.join(args.out, "out0.json")
    with open(out0_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"✅ Summary saved to: {out0_path}")

if __name__ == "__main__":
    main() 