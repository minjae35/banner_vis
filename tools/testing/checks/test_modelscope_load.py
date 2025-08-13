import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
from transformers import Qwen2_5_VLForConditionalGeneration

MODEL_PATH = '/home/intern/banner_vis/pretrained/Qwen2.5-VL-3B-Instruct'

def main():
    print('모델 로딩 중...')
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map='auto'
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f'✅ 모델 파라미터 수: {total_params:,}')

if __name__ == '__main__':
    main() 