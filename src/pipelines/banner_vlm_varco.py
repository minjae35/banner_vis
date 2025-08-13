import os
import sys
import json
import time
import argparse
from PIL import Image
import torch
from transformers import AutoTokenizer

# llava 모듈 경로 추가
sys.path.append('./LLaVA-NeXT')
sys.path.append('../pipelines/LLaVA-NeXT')
sys.path.append('./src/pipelines/LLaVA-NeXT')
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava.mm_utils import tokenizer_image_token, process_images

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
    parser = argparse.ArgumentParser(description='VARCO VLM Detector-Cropped Inference')
    # parser.add_argument('--image', required=True, help='Input image path')
    parser.add_argument('--gpu', type=int, default=0, help='CUDA device index to use')
    args = parser.parse_args()
    # 고정값 할당
    args.image = '/home/intern/banner_vis/src/pipelines/test.jpg'
    args.cfg = '/home/intern/banner_vis/configs/det_config.json'
    args.out = 'output'
    return args

def main():
    args = get_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    img_basename = os.path.splitext(os.path.basename(args.image))[0]
    print("==================== VARCO VLM DETECTOR INTEGRATED PIPELINE ====================")
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

    # 5. Load VLM model & tokenizer
    model_name = "NCSOFT/VARCO-VISION-14B"
    print("[INFO] Loading VLM model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = LlavaQwenForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device,
    )
    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor
    print("[INFO] VLM model loaded.")

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
        # bbox 변환 및 crop (DEBUG print문 삭제)
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

        # Prepare conversation for LLaVA
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

        # Run VLM inference
        print(f"[INFO] Running VLM inference for banner {idx+1}...")
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
        print(f"[INFO] VLM raw output for banner {idx+1}: {decoded}")
        # Try to parse JSON
        try:
            parsed = json.loads(decoded)
            print(f"[INFO] Parsed JSON for banner {idx+1}: {parsed}")
        except Exception as e:
            print(f"[WARNING] Could not parse JSON for banner {idx+1}: {e}")
            parsed = None
        results.append({
            "bbox_normalized": [float(x) for x in norm_bbox],
            "bbox_abs": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(conf),
            "crop_path": crop_path,
            "vlm_raw": decoded,
            "vlm_json": parsed,
        })

    # 7. Save all results
    # output_file = os.path.join(args.out, f"{img_basename}_vlm_results.json")
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)
    # print(f"\n✅ All results saved to: {output_file}")
    # print("==================== SUMMARY ====================")
    # print(f"Total banners detected: {len(results)}")
    # for i, r in enumerate(results):
    #     print(f"  Banner {i+1}: bbox={r['bbox_normalized']}, conf={r['confidence']:.3f}, crop={r['crop_path']}")
    # print("=================================================")

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
    # 이미지별 결과 저장
    out0_path = os.path.join(args.out, f"{img_basename}_out.json")
    with open(out0_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"✅ Summary saved to: {out0_path}")
    
    # 전체 결과 누적 저장
    all_results_path = os.path.join(args.out, "all_results.json")
    all_results = []
    if os.path.exists(all_results_path):
        with open(all_results_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    
    all_results.append(out_json)
    
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"✅ All results accumulated in: {all_results_path}")

if __name__ == "__main__":
    main() 