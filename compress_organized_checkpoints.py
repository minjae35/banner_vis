import tarfile
import os

src_dir = '/home/intern/banner_vis/qwen-tuning/Qwen2.5-VL/qwen-vl-finetune/organized_checkpoints'
output_path = '/home/intern/banner_vis/qwen-tuning/Qwen2.5-VL/qwen-vl-finetune/organized_checkpoints.tar.gz'

with tarfile.open(output_path, 'w:gz') as tar:
    tar.add(src_dir, arcname=os.path.basename(src_dir))

print(f'압축 완료: {output_path}')