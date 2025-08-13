import os
import json
from PIL import Image

# 경로는 본인 환경에 맞게 수정
jsonl_path = "/home/intern/banner_vis/notebooks/output_data.jsonl"
image_root = "/home/intern/banner_vis/data/FlatData/banner_syn_custom/image"

min_pixels = float('inf')
max_pixels = 0

with open(jsonl_path, "r") as f:
    for line in f:
        item = json.loads(line)
        # 단일 이미지만 있다고 가정
        img_name = item.get("image")
        if img_name is None:
            continue
        img_path = os.path.join(image_root, img_name)
        if not os.path.exists(img_path):
            print(f"[MISSING] {img_path}")
            continue
        with Image.open(img_path) as img:
            w, h = img.size
            pixels = w * h
            min_pixels = min(min_pixels, pixels)
            max_pixels = max(max_pixels, pixels)
            print(f"{img_name}: {w}x{h} = {pixels} pixels")
print(f"\n[SUMMARY] min_pixels: {min_pixels}, max_pixels: {max_pixels}")