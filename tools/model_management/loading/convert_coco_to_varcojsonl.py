import os
import json
from glob import glob

LABEL_DIR = "/home/intern/cropped-banner/label"
IMG_DIR = "/home/intern/cropped-banner/image"
OUT_PATH = "varco_finetune_all.jsonl"

cat_id2name = {
    1: "정당 현수막",
    2: "민간 현수막",
    3: "공공 현수막",
    4: "text"
}

cnt_total, cnt_written, cnt_skip = 0, 0, 0
with open(OUT_PATH, "w", encoding="utf-8") as fout:
    for lbl_path in glob(os.path.join(LABEL_DIR, "*.json")):
        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                coco = json.load(f)
        except Exception as e:
            print(f"[ERROR] {lbl_path} JSON 파싱 실패: {e}")
            continue
        for img in coco.get("images", []):
            img_id = img["id"]
            img_file = os.path.join("image", img["file_name"])
            anns = [a for a in coco.get("annotations", []) if a["image_id"] == img_id]
            for ann in anns:
                cnt_total += 1
                text = ann.get("attributes", {}).get("hanguel", "")
                cat = cat_id2name.get(ann.get("category_id", -1), "")
                if not text or not cat or cat == "text":
                    cnt_skip += 1
                    continue
                sample = {
                    "image": img_file,
                    "conversations": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "<obj>이 현수막</obj>의 내용을 꼼꼼하게 읽고, 분류해줘."},
                                {"type": "image"}
                            ]
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": f"현수막 내용: {text}\n분류 결과: {cat}\n판단 이유: [여기에 분류 근거를 입력]"}
                            ]
                        }
                    ]
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                cnt_written += 1
print(f"총 {cnt_total}개 샘플 중 {cnt_written}개를 {OUT_PATH}로 변환/합침 (누락/스킵: {cnt_skip})") 