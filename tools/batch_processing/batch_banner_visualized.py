import os
import sys
import argparse
import glob
from banner_visualized import banner_init_gpu, banner_analysis, banner_release_gpu

def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))

def main():
    parser = argparse.ArgumentParser(description='배너 분석 배치 스크립트')
    parser.add_argument('--input-dir', type=str, required=True, help='이미지 폴더 경로')
    parser.add_argument('--output-dir', type=str, default='./output', help='출력 디렉토리')
    parser.add_argument('--cctv-id', type=int, default=2, help='CCTV ID (기본값: 2)')
    parser.add_argument('--gpu-id', type=int, default=1, help='GPU ID (기본값: 1)')
    parser.add_argument('--config-path', type=str, default='./tmpv5.json', help='설정 파일 경로')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"오류: 입력 폴더 '{args.input_dir}'이 존재하지 않습니다.")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.config_path):
        print(f"오류: 설정 파일 '{args.config_path}'이 존재하지 않습니다.")
        sys.exit(1)

    # 모델 초기화 (한 번만)
    conf_path = args.config_path
    args_model, *ban_models = banner_init_gpu(args.gpu_id, conf_path)

    image_files = sorted([f for f in glob.glob(os.path.join(args.input_dir, '*')) if is_image_file(f)])
    if not image_files:
        print(f"오류: '{args.input_dir}' 폴더에 이미지 파일이 없습니다.")
        sys.exit(1)

    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_img = os.path.join(args.output_dir, f"{base_name}_result.jpg")
        out_json = os.path.join(args.output_dir, f"{base_name}_result.json")
        try:
            banner_analysis(args.cctv_id, img_path, out_img, out_json, ban_models, args.gpu_id)
            print(f"완료: {img_path}")
        except Exception as e:
            print(f"[에러] {img_path} 처리 중 오류 발생: {e}")

    banner_release_gpu(args.gpu_id, ban_models)
    print(f"모든 이미지 처리가 완료되었습니다. 결과는 '{args.output_dir}'에 저장됩니다.")

if __name__ == '__main__':
    main() 