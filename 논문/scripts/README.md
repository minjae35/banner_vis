# Qwen2.5-VL Fine-tuning Scripts

재구성된 파인튜닝 스크립트 구조입니다.

## 📁 폴더 구조

```
scripts/
├── configs/                    # DeepSpeed 설정 파일들
│   ├── zero2.json             # ZeRO Stage 2 설정
│   ├── zero3.json             # ZeRO Stage 3 설정 (기본)
│   └── zero3_offload.json     # ZeRO Stage 3 + CPU Offload 설정
├── experiments/               # 개별 실험 실행 스크립트
│   ├── cot/                   # Chain of Thought 실험들
│   │   ├── run_bal_equal.sh   # BAL-Equal 실험
│   │   ├── run_cw_only.sh     # CW-Only 실험
│   │   ├── run_no_warp.sh     # No-Warp 실험
│   │   └── run_ratio_321.sh   # Ratio 3:2:1 실험
│   └── simple/                # Simple 실험들 (CoT 없음)
│       ├── run_bal_equal.sh   # BAL-Equal Simple 실험
│       ├── run_cw_only.sh     # CW-Only Simple 실험
│       ├── run_no_warp.sh     # No-Warp Simple 실험
│       └── run_ratio_321.sh   # Ratio 3:2:1 Simple 실험
├── batch_scripts/             # 배치 실행 스크립트
│   ├── run_all_experiments.sh      # 모든 CoT 실험 동시 실행 (단계별)
│   ├── run_all_experiments_simple.sh # 모든 Simple 실험 동시 실행 (단계별)
│   ├── run_parallel_cot.sh         # 모든 CoT 실험 병렬 실행
│   ├── run_parallel_simple.sh      # 모든 Simple 실험 병렬 실행
│   ├── run_parallel_all.sh         # 모든 실험 병렬 실행
│   ├── monitor_experiments.sh      # 실험 모니터링 도구
│   └── old_scripts/                # 기존 스크립트들 (백업)
└── templates/                 # 템플릿 스크립트
    └── training_template.sh   # 재사용 가능한 훈련 템플릿
```

## 🚀 사용법

### 1. 개별 실험 실행

#### CoT 실험 (Chain of Thought)
```bash
# BAL-Equal 실험 (GPU 0,1)
./scripts/experiments/cot/run_bal_equal.sh

# CW-Only 실험 (GPU 2,3)
./scripts/experiments/cot/run_cw_only.sh

# No-Warp 실험 (GPU 4,5)
./scripts/experiments/cot/run_no_warp.sh

# Ratio 3:2:1 실험 (GPU 6,7)
./scripts/experiments/cot/run_ratio_321.sh
```

#### Simple 실험 (CoT 없음)
```bash
# BAL-Equal Simple 실험 (GPU 0,1)
./scripts/experiments/simple/run_bal_equal.sh

# CW-Only Simple 실험 (GPU 2,3)
./scripts/experiments/simple/run_cw_only.sh

# No-Warp Simple 실험 (GPU 4,5)
./scripts/experiments/simple/run_no_warp.sh

# Ratio 3:2:1 Simple 실험 (GPU 6,7)
./scripts/experiments/simple/run_ratio_321.sh
```

### 2. 배치 실행

#### 단계별 실행 (GPU 부하 분산)
```bash
# 모든 CoT 실험 동시 실행 (단계별)
./scripts/batch_scripts/run_all_experiments.sh

# 모든 Simple 실험 동시 실행 (단계별)
./scripts/batch_scripts/run_all_experiments_simple.sh
```

#### 병렬 실행 (모든 실험 동시 시작)
```bash
# 모든 CoT 실험 병렬 실행
./scripts/batch_scripts/run_parallel_cot.sh

# 모든 Simple 실험 병렬 실행
./scripts/batch_scripts/run_parallel_simple.sh

# 모든 실험 병렬 실행 (CoT + Simple)
./scripts/batch_scripts/run_parallel_all.sh
```

#### 모니터링 도구
```bash
# 실험 상태 모니터링
./scripts/batch_scripts/monitor_experiments.sh
```

### 3. 템플릿 사용

새로운 실험을 위한 커스텀 스크립트 생성:

```bash
#!/bin/bash

# 실험 설정
export EXPERIMENT_NAME="custom_experiment"
export DATASET_CONFIG="custom_dataset%100"
export GPU_IDS="0,1"
export NUM_GPUS=2
export EPOCHS=25
export LEARNING_RATE=3e-6
export BATCH_SIZE=2
export GRAD_ACCUM_STEPS=4
export USE_SIMPLE="false"

# 템플릿 스크립트 실행
source ./scripts/templates/training_template.sh
```

## 📊 실험 설정

### 실험별 상세 정보

| 실험명 | 데이터 구성 | GPU | 에포크 | 설명 |
|--------|-------------|-----|--------|------|
| BAL-Equal | Crop 2,800 + Flat 2,800 + Warp 2,800 | 0,1 | 20 | 완전 균형 Baseline |
| CW-Only | Crop 2,800 + Warp 2,800 | 2,3 | 30 | Crop + Warp만 사용 |
| No-Warp | Crop 2,800 + Flat 2,800 | 4,5 | 30 | Crop + Flat만 사용 |
| Ratio 3:2:1 | Crop 4,200 + Flat 2,800 + Warp 1,400 | 6,7 | 20 | 3:2:1 비율 |

### CoT vs Simple

- **CoT (Chain of Thought)**: 텍스트 추출 → 추론 과정 → 최종 답안
- **Simple**: 텍스트 추출 → 바로 최종 답안 (CoT 없음)

## ⚙️ 설정 파일

### DeepSpeed 설정

- **zero3.json**: 기본 설정 (ZeRO Stage 3)
- **zero2.json**: 메모리 사용량이 적은 설정
- **zero3_offload.json**: CPU 오프로딩 포함

### 하이퍼파라미터

- **Learning Rate**: 5e-6 (기본)
- **Batch Size**: 2 (GPU당)
- **Gradient Accumulation**: 4
- **실제 배치 크기**: 2 × GPU수 × 4 = 16 (2 GPU 사용 시)

## 📝 로그 및 모니터링

### 로그 파일
- 형식: `train_{experiment_name}_{timestamp}.log`
- 예: `train_bal_equal_20241201_143022.log`

### 모니터링 명령어
```bash
# GPU 사용량 확인
nvidia-smi

# 프로세스 상태 확인
ps aux | grep torchrun

# 로그 실시간 확인
tail -f train_*.log

# 특정 실험 로그 확인
tail -f train_bal_equal_*.log
```

## 🔧 문제 해결

### 일반적인 문제들

1. **GPU 메모리 부족**
   - `batch_size`를 1로 줄이기
   - `grad_accum_steps`를 늘리기

2. **포트 충돌**
   - `MASTER_PORT`를 다른 값으로 설정
   - 스크립트에서 자동으로 랜덤 포트 할당

3. **OMP 경고**
   - `export OMP_NUM_THREADS=1` 설정 (이미 포함됨)

### 디버깅

```bash
# 상세 로그 확인
tail -f train_*.log | grep -E "(ERROR|WARNING|Exception)"

# GPU 메모리 사용량 모니터링
watch -n 1 nvidia-smi

# 프로세스 종료
pkill -f torchrun
```

## 📈 성능 최적화

### 권장 설정

- **8개 GPU 환경**: 4개 실험을 동시에 실행
- **4개 GPU 환경**: 2개 실험을 순차적으로 실행
- **메모리 부족 시**: `zero3_offload.json` 사용

### 예상 실행 시간

- **BAL-Equal/Ratio 3:2:1**: 12-14시간 (20 에포크)
- **CW-Only/No-Warp**: 18-20시간 (30 에포크) 