#!/usr/bin/env python3
"""
중앙 집중식 설정 파일
모든 체크포인트 경로와 설정을 한 곳에서 관리
"""

import os

# 기본 경로 설정
BASE_DIR = "/home/intern2/banner_vis"
DATA_DIR = os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot")

# 모델 설정
MODEL_CONFIGS = {
    "base": {
        "name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "path": "Qwen/Qwen2.5-VL-3B-Instruct",  
        "description": "원본 Qwen2.5-VL-3B-Instruct 모델"
    },
    "pretrained": {
        "name": "Qwen2.5-VL-3B-Instruct",
        "path": os.path.join(BASE_DIR, "pretrained/Qwen2.5-VL-3B-Instruct"),
        "description": "로컬 프리트레인 모델"
    }
}

# 체크포인트 기본 경로
CHECKPOINT_BASE_PATH = "/home/intern2/banner_vis/qwen-tuning/Qwen2.5-VL/qwen-vl-finetune/organized_checkpoints"

# 체크포인트 설정
CHECKPOINT_CONFIGS = {
    "bal_equal": {
        "path": os.path.join(CHECKPOINT_BASE_PATH, "bal_equal/experiment/checkpoints_bal_equal_experiment_gpu01/checkpoint-10500"),
        "gpu_id": 0,
        "description": "Balanced Equal 체크포인트"
    },
    "bal_equal_simple": {
        "path": os.path.join(CHECKPOINT_BASE_PATH, "bal_equal/simple_experiment/checkpoints_bal_equal_simple_experiment_gpu01/checkpoint-10500"),
        "gpu_id": 0,
        "description": "Balanced Equal Simple 체크포인트"
    },
    "c3f2w1": {
        "path": os.path.join(CHECKPOINT_BASE_PATH, "c3f2w1/experiment/checkpoints_c3f2w1_experiment_gpu67/checkpoint-10500"),
        "gpu_id": 1,
        "description": "C3F2W1 체크포인트"
    },
    "c3f2w1_simple": {
        "path": os.path.join(CHECKPOINT_BASE_PATH, "c3f2w1/simple_experiment/checkpoints_c3f2w1_simple_experiment_gpu67/checkpoint-10500"),
        "gpu_id": 1,
        "description": "C3F2W1 Simple 체크포인트"
    },
    "cw_only": {
        "path": os.path.join(CHECKPOINT_BASE_PATH, "cw_only/experiment/checkpoints_cw_only_experiment_gpu23/checkpoint-10500"),
        "gpu_id": 2,
        "description": "CW Only 체크포인트"
    },
    "cw_only_simple": {
        "path": os.path.join(CHECKPOINT_BASE_PATH, "cw_only/simple_experiment/checkpoints_cw_only_simple_experiment_gpu23/checkpoint-10500"),
        "gpu_id": 2,
        "description": "CW Only Simple 체크포인트"
    },
    "no_warp": {
        "path": os.path.join(CHECKPOINT_BASE_PATH, "no_warp/experiment/checkpoints_no_warp_experiment_gpu45/checkpoint-10500"),
        "gpu_id": 3,
        "description": "No Warp 체크포인트"
    },
    "no_warp_simple": {
        "path": os.path.join(CHECKPOINT_BASE_PATH, "no_warp/simple_experiment/checkpoints_no_warp_simple_experiment_gpu45/checkpoint-10500"),
        "gpu_id": 3,
        "description": "No Warp Simple 체크포인트"
    }
}

# 프롬프트 템플릿
PROMPT_TEMPLATES = {
    "standard": (
        "<obj>이 현수막</obj>"
        "위 현수막의 텍스트를 꼼꼼하게 읽고, 내용을 말해주세요. "
        "그 후 이 현수막이 '정당 현수막', '민간 현수막', '공공 현수막' 중 어디에 해당하는지도 함께 판단해주세요.\n\n"
        "판단 기준:\n"
        "1. 정당 현수막: 정당명, 후보자, 선거 관련 문구 포함\n"
        "2. 민간 현수막: 상업 광고, 학원/상점/개인 목적 문구 포함\n"
        "3. 공공 현수막: 정부, 지자체, 공공기관 명칭 또는 정책, 단속, 행정 안내 등 공익적 내용 포함\n\n"
        "출력 형식:\n"
        "- 현수막 내용 : 현수막 내용 읽기\n"
        "- 분류 결과: 정당 현수막 / 민간 현수막 / 공공 현수막\n"
        "- 판단 이유: 텍스트나 문맥을 근거로 분류 이유 설명"
    ),
    "simple": (
        "이 현수막에 적힌 글자를 읽어주세요.\n\n"
        "출력 형식:\n"
        "- 현수막 내용: [읽은 텍스트]\n"
        "- 분류: [정당/민간/공공 현수막]\n"
        "- 판단 이유: [분류 근거]"
    ),
    "ocr_only": (
        "<obj>이 현수막</obj>위 현수막의 텍스트를 꼼꼼하게 읽고, 내용을 말해주세요"
    )
}

# 데이터 파일 경로
DATA_FILES = {
    "test_abs": os.path.join(DATA_DIR, "test_abs_backup.jsonl"),
    "validation": os.path.join(BASE_DIR, "data/validation_test/validation_abs.jsonl"),
    "test_image_wrap": os.path.join(DATA_DIR, "test_image/wrap/test_abs_backup_wrap.jsonl"),
    "test_image_flat": os.path.join(DATA_DIR, "test_image/flat/test_abs_backup_flat.jsonl"),
    "test_image_crop": os.path.join(DATA_DIR, "test_image/crop/test_abs_backup_crop.jsonl"),
    "test_image_all": os.path.join(DATA_DIR, "test_image_all.jsonl")  # 24개 이미지 통합
}

# 테스트 이미지 경로들 (24장)
TEST_IMAGE_PATHS = {
    "wrap": [
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/wrap/D05_240929_bI_IMG_9552_7.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/wrap/D05_240929_bl_IMG_5632_9.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/wrap/C04_240929_Gt_20210130_124631_9.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/wrap/D05_240929_8K_IMG_5533_13.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/wrap/D05_240929_AM_IMG_7629_12.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/wrap/D05_240929_Wx_IMG_4882_9.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/wrap/D05_240929_aj_IMG_5781_17.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/wrap/D05_240929_aq_IMG_0268_13.jpg")
    ],
    "flat": [
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/flat/0742.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/flat/0643.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/flat/0648.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/flat/0661.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/flat/0720.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/flat/0726.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/flat/0727.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/flat/0740.jpg")
    ],
    "crop": [
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/crop/20240911_135829_001_16.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/crop/20240909_174707_15.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/crop/20240910_075423_45.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/crop/20240911_120105_1.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/crop/20240911_121828_18.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/crop/20240911_124914_40.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/crop/20240911_131053_16.jpg"),
        os.path.join(BASE_DIR, "data/experiments/validation_test/datasets/cot/test_image/crop/20240911_132132_16.jpg")
    ]
}

# 결과 디렉토리 기본 경로
RESULTS_BASE_DIR = "/home/intern2/banner_vis/experiments/inference/inference_results"

def get_checkpoint_path(checkpoint_name):
    """체크포인트 경로 반환"""
    if checkpoint_name not in CHECKPOINT_CONFIGS:
        raise ValueError(f"Unknown checkpoint: {checkpoint_name}")
    return CHECKPOINT_CONFIGS[checkpoint_name]["path"]

def get_checkpoint_gpu_id(checkpoint_name):
    """체크포인트 GPU ID 반환"""
    if checkpoint_name not in CHECKPOINT_CONFIGS:
        raise ValueError(f"Unknown checkpoint: {checkpoint_name}")
    return CHECKPOINT_CONFIGS[checkpoint_name]["gpu_id"]

def get_all_checkpoint_names():
    """모든 체크포인트 이름 반환"""
    return list(CHECKPOINT_CONFIGS.keys())

def get_checkpoint_config(checkpoint_name):
    """체크포인트 설정 반환"""
    if checkpoint_name not in CHECKPOINT_CONFIGS:
        raise ValueError(f"Unknown checkpoint: {checkpoint_name}")
    return CHECKPOINT_CONFIGS[checkpoint_name]

def get_model_config(model_type="pretrained"):
    """모델 설정 반환"""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_CONFIGS[model_type]

def get_model_path(model_type="pretrained"):
    """모델 경로 반환"""
    return get_model_config(model_type)["path"]

def get_model_name(model_type="pretrained"):
    """모델 이름 반환"""
    return get_model_config(model_type)["name"]

def get_test_images(folder_type="all"):            # warp, flat, crop 각각 8장씩 (총 24장)
    """테스트 이미지 경로들 반환"""
    if folder_type == "all":
        # 모든 폴더의 이미지들을 하나의 리스트로 합치기
        all_images = []
        for folder_images in TEST_IMAGE_PATHS.values():
            all_images.extend(folder_images)
        return all_images
    elif folder_type in TEST_IMAGE_PATHS:
        return TEST_IMAGE_PATHS[folder_type]
    else:
        raise ValueError(f"Unknown folder type: {folder_type}")

def get_test_images_count(folder_type="all"):
    """테스트 이미지 개수 반환"""
    return len(get_test_images(folder_type)) 