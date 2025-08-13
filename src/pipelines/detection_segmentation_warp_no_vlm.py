import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from banner_visualized import load_sam_model, load_det_model, banDet, banSeg, preprocess_image, extract_center, select_images_based_on_pred_points, ResizeLongestSide, banWarp, find_four_corners, order_points, is_long_banner_horizontal, DeNormalize
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2
from PIL import Image

# 출력 폴더 생성
os.makedirs('seg_test_output', exist_ok=True)
os.makedirs('seg_test_output/warp_steps', exist_ok=True)

with open('/home/intern/banner_vis/configs/tmpv5.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
sam_model_path = config['sam']['model_path']
det_model_path = config['det']['model_path']
gpu_id = config['sam']['gpu_id']

sam_model = load_sam_model(gpu_id, sam_model_path)
det_model, test_pipeline = load_det_model(gpu_id, det_model_path)

img_path = '/home/intern/banner_vis/src/pipelines/test.jpg'




# 1. 원본 이미지 로드 및 비율 유지 리사이즈+패딩
img = Image.open(img_path).convert('RGB')
img_np = np.array(img)
original_size = img_np.shape[:2]

# 2. Detection
class DummyDetection:
    def __init__(self):
        self.conf_thres = 0.3
        self.save_results = False
        self.out_img_filename = 'dummy.png'
        self.save_root_path = './output'
class DummyParams:
    def __init__(self):
        self.detection = DummyDetection()
params = DummyParams()
data = [{'width': original_size[1], 'height': original_size[0]}]

# Detection은 원본 이미지로 수행
imgs_array = [img_np]
imgs = torch.from_numpy(img_np.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
imgs = imgs.to(f'cuda:{gpu_id}')

det_results = banDet(det_model, test_pipeline, imgs_array, imgs, params, data)

# 3. Detection 결과에서 bbox 추출 및 crop+padding
if hasattr(det_results, 'cpu'):
    det_results_np = det_results.cpu().numpy()
else:
    det_results_np = det_results.numpy() if hasattr(det_results, 'numpy') else det_results
if len(det_results_np.shape) == 3:
    det_results_np = det_results_np[0]  # (N, 6)

cropped_imgs = []
crop_boxes = []
resize_params = []
for box in det_results_np:
    if len(box) < 4:
        continue
    x1, y1, x2, y2 = map(int, box[:4])
    cropped = img_np[y1:y2, x1:x2, :]
    h, w = cropped.shape[:2]
    scale = 1024 / max(h, w)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = 1024 - new_h
    pad_w = 1024 - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    cropped_imgs.append(padded)
    crop_boxes.append((x1, y1, x2, y2))
    resize_params.append((scale, top, left, new_h, new_w, h, w))

if not cropped_imgs:
    print("No detection found!")
    exit()

# 4. Segmentation 입력 준비
seg_input = []
for padded in cropped_imgs:
    img_tensor = padded.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).contiguous()
    seg_input.append(img_tensor)
seg_input = torch.cat(seg_input, dim=0).to(f'cuda:{gpu_id}')

# Detection bbox 중심점 계산 (crop 좌표 기준)
pred_points = []
for i, (x1, y1, x2, y2) in enumerate(crop_boxes):
    crop_w, crop_h = x2-x1, y2-y1
    cx, cy = crop_w//2, crop_h//2
    # 패딩/리사이즈 후 중심점 좌표로 변환
    scale, top, left, new_h, new_w, h, w = resize_params[i]
    cx_resized = int(cx * scale) + left
    cy_resized = int(cy * scale) + top
    pred_points.append([i, 1.0, cx_resized, cy_resized])

selected_imgs, selected_points = select_images_based_on_pred_points(seg_input, pred_points)
selected_imgs = selected_imgs.to(f'cuda:{gpu_id}')
selected_points = selected_points.to(f'cuda:{gpu_id}')

# 디버깅: 각 리스트/텐서 shape/개수 출력
print('num det_results_np:', len(det_results_np))
print('num crop_boxes:', len(crop_boxes))
print('num cropped_imgs:', len(cropped_imgs))
print('num resize_params:', len(resize_params))
print('seg_input 준비 shape:', [x.shape for x in cropped_imgs])
print('pred_points:', pred_points)

# Segmentation 입력 준비 (명확한 for문 구조)
seg_results = []
seg_scores = []
all_masks = []
for i, (padded, (x1, y1, x2, y2), (scale, top, left, new_h, new_w, h, w)) in enumerate(zip(cropped_imgs, crop_boxes, resize_params)):
    img_tensor = padded.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0).contiguous().to(f'cuda:{gpu_id}')
    # 중심점 계산 (crop 기준 → 패딩/리사이즈 변환)
    crop_w, crop_h = x2-x1, y2-y1
    cx, cy = crop_w//2, crop_h//2
    cx_resized = int(cx * scale) + left
    cy_resized = int(cy * scale) + top
    point = torch.tensor([[[cx_resized, cy_resized]]], dtype=torch.float32).to(f'cuda:{gpu_id}')
    # Segmentation 파라미터
    class DummySegmentation:
        def __init__(self, mask_threshold=0.5, save_results=False):
            self.mask_threshold = mask_threshold
            self.save_results = save_results
    class DummyParamsSeg:
        def __init__(self):
            self.segmentation = DummySegmentation()
    params_seg = DummyParamsSeg()
    # Segmentation 실행
    seg_result, scores = banSeg(sam_model, img_tensor, point, [0], [0], params_seg)
    seg_results.append(seg_result[0])
    seg_scores.append(scores[0])
    # 결과 저장 및 시각화 (기존 코드 활용)
    mask = seg_result[0]
    if hasattr(mask, "cpu"):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    mask_img = (mask_np * 255).astype(np.uint8)
    # 1. 크롭된 현수막 이미지 저장
    cv2.imwrite(f'seg_test_output/crop_{i}.png', cv2.cvtColor(cropped_imgs[i], cv2.COLOR_RGB2BGR))
    # 2. 패딩+리사이즈된 현수막 이미지 저장
    cv2.imwrite(f'seg_test_output/padded_{i}.png', cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))
    # 3. 패딩 제거 및 원래 크기로 역리사이즈
    mask_unpadded = mask_img[top:top+new_h, left:left+new_w]
    mask_orig_crop = cv2.resize(mask_unpadded, (w, h), interpolation=cv2.INTER_NEAREST)
    # 4. 원본 이미지 크기만큼 빈 마스크 생성 후, bbox 위치에 붙이기
    mask_full = np.zeros((original_size[0], original_size[1]), dtype=np.uint8)
    mask_full[y1:y2, x1:x2] = mask_orig_crop
    # 5. 마스크 저장
    cv2.imwrite(f'seg_test_output/seg_mask_{i}_origsize.png', mask_full)
    all_masks.append(mask_full)
    # 6. 시각화(overlay)
    img_raw_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    color_mask = cv2.applyColorMap(mask_full, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_raw_bgr, 0.7, color_mask, 0.3, 0)
    cv2.imwrite(f'seg_test_output/seg_overlay_{i}.jpg', overlay)
    print(f'Saved: crop_{i}.png, padded_{i}.png, seg_mask_{i}_origsize.png, seg_overlay_{i}.jpg')

# 여러 현수막 마스크를 합쳐서 전체 마스크로 저장
if all_masks:
    final_mask = np.clip(np.sum(all_masks, axis=0), 0, 255).astype(np.uint8)
    cv2.imwrite('seg_test_output/seg_mask_all.png', final_mask)
    print('Saved: seg_mask_all.png')

# Detection 결과 시각화 및 저장 (기존 코드)
img_det_vis = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR).copy()
for box in det_results_np:
    if len(box) < 4:
        continue
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(img_det_vis, (x1, y1), (x2, y2), (0,255,0), 2)
cv2.imwrite('seg_test_output/detection_result.jpg', img_det_vis)
print('Saved: detection_result.jpg')

# ===== WARPING 단계 추가 =====
print("\n=== WARPING 단계 시작 ===")

# Warping을 위한 파라미터 설정
class DummyWarping:
    def __init__(self):
        self.save_results = True
class DummyParamsWarp:
    def __init__(self):
        self.warping = DummyWarping()
        self.save_root_path = './seg_test_output'
        self.out_img_filename = 'warp_test.jpg'  # .jpg 확장자 추가
params_warp = DummyParamsWarp()

# Warping 실행
warp_out, inverse_transform_matrix, four_corners, warped_idxes = banWarp(
    seg_results, selected_imgs, pred_points, params_warp)

print(f"Warping 완료: {len(warp_out)}개 현수막 처리됨")

# Warping 결과 시각화
for i, (warped_img, corners) in enumerate(zip(warp_out, four_corners)):
    # 1. 펴진 현수막 이미지 저장
    cv2.imwrite(f'seg_test_output/warped_banner_{i}.jpg', cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR))
    
    # 2. 원본 이미지에 모서리 표시
    img_with_corners = img_np.copy()
    for j, corner in enumerate(corners):
        cv2.circle(img_with_corners, tuple(corner), 10, (255, 0, 0), -1)  # 파란색 원
        cv2.putText(img_with_corners, str(j), tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 모서리들을 선으로 연결
    for j in range(4):
        pt1 = tuple(corners[j])
        pt2 = tuple(corners[(j + 1) % 4])
        cv2.line(img_with_corners, pt1, pt2, (0, 255, 255), 3)  # 노란색 선
    
    cv2.imwrite(f'seg_test_output/corners_{i}.jpg', cv2.cvtColor(img_with_corners, cv2.COLOR_RGB2BGR))
    
    # 3. 합성 이미지 (원본 + 모서리 + 펴진 결과)
    h, w = img_np.shape[:2]
    composite = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    # 원본 이미지
    composite[:, :w] = img_np
    
    # 모서리 표시
    composite[:, w:w*2] = img_with_corners
    
    # 펴진 결과 (크기 조정)
    warped_resized = cv2.resize(warped_img, (w, h))
    composite[:, w*2:] = warped_resized
    
    cv2.imwrite(f'seg_test_output/composite_{i}.jpg', cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    
    print(f'Warping 결과 {i}: warped_banner_{i}.jpg, corners_{i}.jpg, composite_{i}.jpg')

print("\n=== 모든 단계 완료 ===")
print("결과 파일들이 seg_test_output/ 폴더에 저장되었습니다.")
print("- detection_result.jpg: 검출 결과")
print("- seg_mask_*_origsize.png: 세그멘테이션 마스크")
print("- seg_overlay_*.jpg: 세그멘테이션 오버레이")
print("- warped_banner_*.jpg: 펴진 현수막")
print("- corners_*.jpg: 모서리 검출")
print("- composite_*.jpg: 전체 과정 합성")
print("- warp_steps/ 폴더: 각 단계별 상세 결과")