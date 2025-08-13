import os
from PIL import Image

padded_root = "/home/intern/banner_vis/data/FlatData/banner_syn_custom/image_padded"

not_padded = []

for fname in os.listdir(padded_root):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    path = os.path.join(padded_root, fname)
    with Image.open(path) as img:
        w, h = img.size
        if w % 28 != 0 or h % 28 != 0:
            print(f"[NOT PADDED] {fname}: {w}x{h}")
            not_padded.append((fname, w, h))
        else:
            print(f"[OK] {fname}: {w}x{h}")

print(f"\n[SUMMARY] 총 {len(not_padded)}개 이미지가 28의 배수가 아닙니다.")
if not_padded:
    print("패딩이 안 된 이미지 목록:")
    for fname, w, h in not_padded:
        print(f"  {fname}: {w}x{h}")
else:
    print("모든 이미지가 28의 배수로 패딩되었습니다.") 