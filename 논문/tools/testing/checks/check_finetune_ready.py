import os
import json

# 1. 파일 경로
JSONL_PATH = 'varco_ocr_finetune.jsonl'
IMAGE_ROOT = '/home/intern/cropped-banner/image'

print("==== 데이터셋/환경 체크리스트 ====")

# 2. jsonl 파일 열기
try:
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"[✅] {JSONL_PATH} 파일 열기 성공 ({len(lines)}개 샘플)")
except Exception as e:
    print(f"[❌] {JSONL_PATH} 파일 열기 실패: {e}")
    exit(1)

# 3. 샘플 5개 구조 확인
ok = True
for i, line in enumerate(lines[:5]):
    try:
        obj = json.loads(line)
        assert 'image' in obj, "image 키 없음"
        assert 'conversations' in obj, "conversations 키 없음"
        assert isinstance(obj['conversations'], list), "conversations가 리스트 아님"
        assert any(conv['role'] == 'assistant' for conv in obj['conversations']), "assistant 없음"
        assert any(conv['role'] == 'user' for conv in obj['conversations']), "user 없음"
    except Exception as e:
        print(f"[❌] 샘플 {i+1} 구조 오류: {e}")
        ok = False
if ok:
    print("[✅] 샘플 5개 구조 정상")

# 4. 이미지 파일 실제 존재 여부 (샘플 5개)
ok = True
for i, line in enumerate(lines[:5]):
    obj = json.loads(line)
    img_path = obj['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(IMAGE_ROOT, os.path.basename(img_path))
    if not os.path.exists(img_path):
        print(f"[❌] 이미지 파일 없음: {img_path}")
        ok = False
if ok:
    print("[✅] 샘플 5개 이미지 파일 존재")

# 5. 필수 패키지 임포트
try:
    import torch
    import transformers
    import peft
    from PIL import Image
    print("[✅] 필수 패키지 임포트 성공")
except Exception as e:
    print(f"[❌] 필수 패키지 임포트 실패: {e}")

# 6. GPU 사용 가능 여부
try:
    if torch.cuda.is_available():
        print(f"[✅] GPU 사용 가능 (CUDA: {torch.cuda.get_device_name(0)})")
    else:
        print("[⚠️] GPU 사용 불가 (CPU만 사용)")
except Exception as e:
    print(f"[❌] GPU 체크 실패: {e}")

print("==== 체크리스트 점검 완료 ====") 