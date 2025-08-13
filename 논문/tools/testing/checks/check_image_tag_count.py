import json

jsonl_path = "/home/intern/banner_vis/notebooks/output_data.jsonl"
multi_image_samples = 0
max_image_tags = 0
sample_ids = []

with open(jsonl_path, "r") as f:
    for line in f:
        item = json.loads(line)
        count = 0
        for conv in item.get("conversations", []):
            if "<image>" in conv.get("value", ""):
                count += conv["value"].count("<image>")
        if count > 1:
            multi_image_samples += 1
            max_image_tags = max(max_image_tags, count)
            sample_ids.append(item.get("id", "(no id)"))

print(f"여러 <image> 태그가 있는 샘플: {multi_image_samples}개, 최대 태그 수: {max_image_tags}")
if sample_ids:
    print("예시 샘플 id:", sample_ids[:10]) 