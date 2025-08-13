import os
import sys
import json
import time
import argparse
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

# Detector imports (from banner_visualized.py)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from banner_visualized import load_det_model, preprocess_image, banDet

def crop_bbox(image, bbox):
    # bbox: [x1, y1, x2, y2] (normalized)
    width, height = image.size
    x1 = int(bbox[0] * width)
    y1 = int(bbox[1] * height)
    x2 = int(bbox[2] * width)
    y2 = int(bbox[3] * height)
    return image.crop((x1, y1, x2, y2))

def get_args():
    parser = argparse.ArgumentParser(description='VARCO VLM Detector-Cropped Inference with SmolVLM')
    # parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--gpu', type=int, default=7, help='CUDA device index to use (e.g. 6 or 7)')
    args = parser.parse_args()
    args.image = '/home/intern/banner_vis/src/pipelines/test.jpg'
    args.cfg = '/home/intern/banner_vis/configs/det_config.json'
    args.out = 'output'
    return args

def main():
    args = get_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    img_basename = os.path.splitext(os.path.basename(args.image))[0]

    print("=== VARCO VLM DETECTOR + SmolVLM PIPELINE ===")
    print(f" Image:      {args.image}")
    print(f" Config:     {args.cfg}")
    print(f" Output dir: {args.out}")
    print(f" Device:     {device}\n")

    # 1. Load image
    image = Image.open(args.image).convert("RGB")
    width, height = image.size
    print(f"[INFO] Image loaded: {image.size} (W x H)")

    # 2. Load detector model
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    det_model_path = cfg["det"]["model_path"]
    print("[INFO] Loading detector model...")
    det_model, test_pipeline = load_det_model(args.gpu, det_model_path)
    print("[INFO] Detector model loaded.")

    # 3. Preprocess for detection
    from easydict import EasyDict
    import banner_visualized
    params = EasyDict(banner_visualized.config.params)
    params.save_root_path = args.out
    params.save_path = img_basename
    params.out_img_filename = f"{img_basename}_detected.jpg"
    imgs, ratio, orig_size, imgs_array = preprocess_image(
        args.image, params, device=device
    )
    print(f"[INFO] Preprocessed image for detection. imgs.shape: {imgs.shape}")

    # 4. Run detection
    data = [{
        "video_name": args.image,
        "frame_id": i,
        "width": orig_size[1],
        "height": orig_size[0],
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
    print(f"[INFO] {det_results.shape[1]} banner(s) detected.\n")

    # 5. Load SmolVLM processor & model
    model_name = "HuggingFaceTB/SmolVLM-Base"
    print("[INFO] Loading SmolVLM model and processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).to(device)
    print("[INFO] SmolVLM model & processor loaded.\n")

    # 6. For each bbox, crop and run SmolVLM inference
    prompt_text = (
        "이 현수막의 내용을 읽고 분류해줘.\n"
        "그 후 이 현수막이 '정당 현수막', '민간 현수막', '공공 현수막' 중 어디에 해당하는지도 판단해주세요.\n\n"
        "판단 기준:\n"
        "1. 정당 현수막: 정당명, 후보자, 선거 관련 문구 포함\n"
        "2. 민간 현수막: 상업 광고, 학원/상점/개인 목적 문구 포함\n"
        "3. 공공 현수막: 정부, 지자체, 공공기관 명칭 또는 정책, 단속, 행정 안내 등 공익적 내용 포함\n\n"
        "출력 형식:\n"
        "- 현수막 내용 : 현수막 내용 읽기\n"
        "- 분류 결과: 정당 현수막 / 민간 현수막 / 공공 현수막\n"
        "- 판단 이유: 텍스트나 문맥을 근거로 분류 이유 설명"
    )

    results = []
    for idx, det in enumerate(det_results[0]):
        if len(det) < 5:
            continue
        x1, y1, x2, y2, conf = det[:5]
        sx, sy = width / 1024, height / 1024
        x1o, y1o = float(x1)*sx, float(y1)*sy
        x2o, y2o = float(x2)*sx, float(y2)*sy
        norm_bbox = [x1o/width, y1o/height, x2o/width, y2o/height]
        cropped = crop_bbox(image, norm_bbox)
        crop_path = os.path.join(args.out, f"{img_basename}_crop_{idx+1}.jpg")
        cropped.save(crop_path)
        print(f"[INFO] Cropped banner saved: {crop_path} (size: {cropped.size})")

        # 공식 예시 방식으로 프롬프트 및 입력 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        print(f"[DEBUG] Prompt for banner {idx+1}:\n{prompt}")
        inputs = processor(text=prompt, images=[cropped], return_tensors="pt")
        inputs = inputs.to(device)

        # inference
        print(f"[INFO] Running SmolVLM inference for banner {idx+1}...")
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True,
            )
        decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        print(f"[INFO] SmolVLM output for banner {idx+1}: {decoded}\n")

        results.append({
            "bbox_normalized": norm_bbox,
            "bbox_abs": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(conf),
            "crop_path": crop_path,
            "vlm_raw": decoded,
        })

    # 7. Post-process & save JSON
    from datetime import datetime
    annotation = []
    summary = {}
    for idx, r in enumerate(results):
        ori_txt, ban_cls, reason = "", "", ""
        for line in r["vlm_raw"].splitlines():
            if "현수막 내용" in line:
                ori_txt = line.split(":",1)[-1].strip()
            if "분류 결과" in line:
                ban_cls = line.split(":",1)[-1].strip()
            if "판단 이유" in line:
                reason = line.split(":",1)[-1].strip()
        annotation.append({
            "bbox_normalized": r["bbox_normalized"],
            "bbox_abs": r["bbox_abs"],
            "confidence": r["confidence"],
            "crop_path": r["crop_path"],
        })
        summary[str(idx)] = {"ori_txt": ori_txt, "ban_cls": ban_cls, "reason": reason}

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
