
< Notion link >
[Description page](https://guttural-goose-f4d.notion.site/Qwen25-VL-finetune-24813c56ed7a8014a5f4d9e8b5ac432e)



< 1단계: 실험 유형 선택 >
**bal_equal**: 균형잡힌 데이터셋
**c3f2w1**: 특정 비율의 데이터셋
**cw_only**: 특정 조건만의 데이터셋
**no_warp**: 왜곡 없는 데이터셋

< 2단계: 학습 방식 선택 >
**simgle**: 간단한 분류 학습
**cot**: Chain-of-Thought(단계별 사고) 학습

< 3단계: 스크립트 실행 >
```bash
cd scripts/experiments/simple/
bash run_bal_equal.sh

cd scripts/experiments/cot/
bash run_c3f2w1.sh
```

< 4단계: 일괄 시행 (모든 실험) >
```bash
# 모든 실험을 한번에 실행
cd scripts/batch_scripts/
bash run_all_experiments.sh
```