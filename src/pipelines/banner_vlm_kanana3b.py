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
    width, height = image.size
    x1 = int(bbox[0] * width)
    y1 = int(bbox[1] * height)

    x2 = int(bbox[2] * width)
    y2 = int(bbox[3] * height)
    return image.crop((x1, y1, x2, y2))

def get_args():
    parser = argparse.ArgumentParser(description='Kanana-1.5-V-3B Detector-Cropped Inference')
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
    # Use a simpler prompt for better model compatibility
    prompt_text = "이 현수막의 내용을 읽어줘."
    # Only process the first detected crop for single-image test
    if det_results[0].shape[0] > 0:
        det = det_results[0][0]
        if len(det) >= 5:
            x1, y1, x2, y2, conf = det[:5]
            sx, sy = width / 1024, height / 1024
            x1o, y1o = float(x1)*sx, float(y1)*sy
            x2o, y2o = float(x2)*sx, float(y2)*sy
            norm_bbox = [x1o/width, y1o/height, x2o/width, y2o/height]
            cropped = crop_bbox(image, norm_bbox)
            crop_path = os.path.join(img_outdir, f"crop_1.jpg")
            cropped.save(crop_path)

            # Prepare sample for batch_encode_collate
            conv = [
                {"role": "system", "content": "The following is a conversation between a curious human and AI assistant."},
                {"role": "user", "content": "<image>"},
                {"role": "user", "content": prompt_text}
            ]
            sample = {
                "image": [cropped],
                "conv": conv
            }
            batch = [sample]
            inputs = processor.batch_encode_collate(
                batch, padding_side="left", add_generation_prompt=True, max_length=8192
            )
            inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            # Convert all tensors to float16 to match model weights
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if v.dtype in [torch.float32, torch.float64]:
                        inputs[k] = v.half()
                    elif v.dtype == torch.int64:
                        inputs[k] = v.long()

            gen_kwargs = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.95,
                "num_beams": 1,
                "do_sample": True,
            }
            try:
                with torch.no_grad():
                    gens = model.generate(**inputs, **gen_kwargs)
                    text_outputs = processor.tokenizer.batch_decode(gens, skip_special_tokens=True)
                    print("Decoded:", text_outputs[0])
            except Exception as e:
                print(f"[ERROR] model.generate failed for {img_path} (crop 1): {e}")
                return None
            results = [{
                "bbox_normalized": norm_bbox,
                "bbox_abs": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": float(conf),
                "crop_path": crop_path,
                "vlm_raw": text_outputs[0],
            }]
    else:
        print(f"[WARNING] No valid crops detected in {img_path}!")
        return None

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
    # 5. Load Kanana-1.5-V-3B model & processor with trust_remote_code=True
    model_name = "kakaocorp/kanana-1.5-v-3b-instruct"
    print("[INFO] Loading Kanana-1.5-V-3B model and processor...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True  # This is the crucial change
    )
    model.to(device)
    model = model.half()  # Ensure model is in float16
    print("[INFO] Kanana-1.5-V-3B model loaded.")
    all_results = []
    # 단일 이미지 처리
    out = process_image(args.image, args, device, det_model, test_pipeline, processor, model)
    if out is not None:
        with open(os.path.join(args.out, "vlm_results.json"), "w", encoding="utf-8") as f:
            json.dump([out], f, ensure_ascii=False, indent=2)
        print(f"✅ Result saved to: {os.path.join(args.out, 'vlm_results.json')}")

if __name__ == "__main__":
    main()
