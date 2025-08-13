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
    parser = argparse.ArgumentParser(description='Qwen2.5-VL-3B Detector-Cropped Inference')
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument('--image', help='Input image path')
    # group.add_argument('--image_dir', help='Input image directory (process all jpg/png)')
    parser.add_argument('--gpu', type=int, default=7, help='CUDA device index to use (e.g. 6 or 7)')
    args = parser.parse_args()
    args.image = '/home/intern/banner_vis/src/pipelines/test.jpg'
    args.cfg = '/home/intern/banner_vis/configs/det_config.json'
    args.out = 'output'
    return args


def process_image(img_path, args, device, det_model, test_pipeline, processor, model):
    img_basename = os.path.splitext(os.path.basename(img_path))[0]
    img_outdir = os.path.join(args.out, img_basename)
    os.makedirs(img_outdir, exist_ok=True)
    image = Image.open(img_path).convert("RGB")
    width, height = image.size
    # 3. Preprocess image for detection
    from easydict import EasyDict
    import banner_visualized
    params = EasyDict(banner_visualized.config.params)
    params.save_root_path = img_outdir
    params.save_path = img_basename
    params.out_img_filename = f"{img_basename}_detected.jpg"
    imgs, ratio, orig_size, imgs_array = preprocess_image(
        img_path, params, device=device
    )
    # 4. Run detection
    data = [{
        "video_name": img_path,
        "frame_id": i,
        "width": orig_size[1],
        "height": orig_size[0],
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "annotation": [],
        "banner_annotation": [],
    } for i in range(imgs.shape[0])]
    det_results = banDet(det_model, test_pipeline, imgs_array, imgs, params, data)
    if det_results.numel() == 0 or det_results.shape[1] == 0:
        print(f"[WARNING] No banners detected in {img_path}! Skipping.")
        return None
    results = []
    prompt_text = (
        "<obj>이 현수막</obj>"
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
    for idx, det in enumerate(det_results[0]):
        if len(det) < 5:
            continue
        x1, y1, x2, y2, conf = det[:5]
        sx, sy = width / 1024, height / 1024
        x1o, y1o = float(x1)*sx, float(y1)*sy
        x2o, y2o = float(x2)*sx, float(y2)*sy
        norm_bbox = [x1o/width, y1o/height, x2o/width, y2o/height]
        cropped = crop_bbox(image, norm_bbox)
        crop_path = os.path.join(img_outdir, f"crop_{idx+1}.jpg")
        cropped.save(crop_path)
        messages = [
            {"role": "system", "content": "You are a helpful assistant that reads Korean text in images and classifies banners into 정당/민간/공공 현수막."},
            {"role": "user", "content": [
                {"type": "image", "image": crop_path},
                {"type": "text", "text": prompt_text},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        vision_inputs = processor(text=[text], images=[cropped], padding=True, return_tensors="pt")
        vision_inputs = {k: v.to(device) for k, v in vision_inputs.items()}
        try:
            with torch.no_grad():
                gen_ids = model.generate(**vision_inputs, max_new_tokens=512)
                gen_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(vision_inputs["input_ids"], gen_ids)
                ]
                decoded = processor.batch_decode(
                    gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
        except Exception as e:
            print(f"[ERROR] model.generate failed for {img_path} (crop {idx+1}): {e}")
            for k, v in vision_inputs.items():
                print(f"{k}: shape={v.shape}, dtype={v.dtype}, device={v.device}")
                print(f"  min={v.min().item()}, max={v.max().item()}, mean={v.float().mean().item()}")
            continue
        results.append({
            "bbox_normalized": norm_bbox,
            "bbox_abs": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(conf),
            "crop_path": crop_path,
            "vlm_raw": decoded,
        })
    # Post-process & save per-image results
    from datetime import datetime
    annotation, summary = [], {}
    for i, r in enumerate(results):
        ori_txt = ban_cls = reason = ""
        for line in r["vlm_raw"].splitlines():
            if "현수막 내용" in line: ori_txt = line.split(":",1)[-1].strip()
            if "분류 결과"   in line: ban_cls = line.split(":",1)[-1].strip()
            if "판단 이유"   in line: reason = line.split(":",1)[-1].strip()
        annotation.append({
            "bbox_normalized": r["bbox_normalized"],
            "bbox_abs":         r["bbox_abs"],
            "confidence":       r["confidence"],
            "crop_path":        r["crop_path"]
        })
        summary[str(i)] = {"ori_txt": ori_txt, "ban_cls": ban_cls, "reason": reason}
    out = {
        "image_name":       img_path,
        "width":            width,
        "height":           height,
        "time":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "annotation":       annotation,
        "banner_annotation": [],
        "summary":          summary
    }
    out_path = os.path.join(img_outdir, "out.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✅ Summary saved to: {out_path}")
    return out


def main():
    args = get_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    # 2. Load detector model
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    det_model_path = cfg["det"]["model_path"]
    print("[INFO] Loading detector model...")
    det_model, test_pipeline = load_det_model(args.gpu, det_model_path)
    print("[INFO] Detector model loaded.")
    # 5. Load Qwen2.5-VL-3B model & processor
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    print("[INFO] Loading Qwen2.5-VL-3B model and processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True
    )
    model.to(device)
    print("[INFO] Qwen2.5-VL-3B model loaded.")
    all_results = []
    if args.image_dir:
        # 지정된 10개 파일명만 처리
        base_names = [
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
        image_list = [os.path.join(args.image_dir, f+".jpg") for f in base_names]
        print(f"[INFO] Processing only 10 specified images:")
        for img_path in image_list:
            print(f"  {img_path}")
        for img_path in image_list:
            if not os.path.exists(img_path):
                print(f"[WARNING] File not found: {img_path}")
                continue
            print(f"\n=== Processing: {img_path} ===")
            out = process_image(img_path, args, device, det_model, test_pipeline, processor, model)
            if out is not None:
                all_results.append(out)
        # Save all results
        with open(os.path.join(args.out, "vlm_results.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"✅ All results saved to: {os.path.join(args.out, 'vlm_results.json')}")
    else:
        out = process_image(args.image, args, device, det_model, test_pipeline, processor, model)
        if out is not None:
            with open(os.path.join(args.out, "vlm_results.json"), "w", encoding="utf-8") as f:
                json.dump([out], f, ensure_ascii=False, indent=2)
            print(f"✅ Result saved to: {os.path.join(args.out, 'vlm_results.json')}")

if __name__ == "__main__":
    main()
