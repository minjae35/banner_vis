#!/usr/bin/env python3
import os
import cv2
import numpy as np
from PIL import Image
import glob

def resize_with_padding(image_path, output_path, target_size=(672, 672), padding_color=(255, 255, 255)):
    """
    이미지를 target_size로 리사이즈하면서 현수막이 잘리지 않도록 살짝 줄이고 흰색 패딩 추가
    """
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: 이미지를 로드할 수 없습니다: {image_path}")
        return False
    
    h, w = img.shape[:2]
    target_w, target_h = target_size
    
    # 현수막이 잘리지 않도록 살짝 줄이기 (0.9배)
    scale_factor = 0.9
    
    # 새로운 크기 계산
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    
    # 이미지 리사이즈 (현수막이 잘리지 않도록)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 패딩 계산 (중앙 정렬)
    pad_w = max(0, target_w - new_w)
    pad_h = max(0, target_h - new_h)
    
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    # 흰색 패딩 추가
    padded = cv2.copyMakeBorder(
        resized, 
        pad_top, pad_bottom, pad_left, pad_right, 
        cv2.BORDER_CONSTANT, 
        value=padding_color
    )
    
    # 최종 크기 확인 및 조정
    if padded.shape[0] != target_h or padded.shape[1] != target_w:
        padded = cv2.resize(padded, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # 저장
    cv2.imwrite(output_path, padded)
    return True

def main():
    # 경로 설정
    input_dir = "/home/intern/banner_vis/data/crop_7000/images"
    output_dir = "/home/intern/banner_vis/data/crop_7000/images_resized"
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Crop 이미지 672x672 리사이즈 (패딩 포함) ===")
    
    # 이미지 파일 목록
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    print(f"총 이미지 수: {len(image_files)}")
    
    # 처리
    success_count = 0
    for i, image_path in enumerate(image_files):
        if i % 100 == 0:
            print(f"처리 중... {i}/{len(image_files)}")
        
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, filename)
        
        if resize_with_padding(image_path, output_path):
            success_count += 1
        else:
            print(f"실패: {filename}")
    
    print(f"\n=== 완료 ===")
    print(f"성공: {success_count}개")
    print(f"실패: {len(image_files) - success_count}개")
    print(f"출력 디렉토리: {output_dir}")
    
    # 샘플 이미지 크기 확인
    if success_count > 0:
        sample_files = glob.glob(os.path.join(output_dir, "*.jpg"))[:3]
        for sample_file in sample_files:
            img = cv2.imread(sample_file)
            print(f"샘플 {os.path.basename(sample_file)}: {img.shape}")

if __name__ == "__main__":
    main() 