# LSTM 마커 위치 예측 모델 학습

## 개요

이 프로젝트는 공압 제어 신호(압력 센서 데이터)를 입력으로 받아 소프트 로봇의 마커 위치를 예측하는 LSTM 모델을 학습합니다.

- **중심 마커**: marker_2 (원점)
- **예측 마커**: marker_3, marker_4, marker_5, marker_6, marker_7 (중심 마커 대비 상대 위치)

## 설치

필요한 패키지 설치:

```bash
pip install torch numpy pandas scikit-learn tqdm matplotlib seaborn
pip install optuna plotly kaleido  # hyperparameter tuning용 (선택)
```

## 기본 사용법

### 1. 기본 학습 실행

```bash
cd /home/min/project_SORO/src/lstm/lstm
python3 train_lstm_markers_test.py
```

기본 설정으로 학습이 진행되며, 결과는 `./runs/lstm_YYYYMMDD-HHMMSS/` 폴더에 저장됩니다.

### 2. 커스텀 파라미터로 학습

```bash
python3 train_lstm_markers_test.py \
  --window 80 \
  --horizon 2 \
  --epochs 50 \
  --batch_size 512 \
  --lr 0.0005 \
  --hidden_size 256 \
  --num_layers 3 \
  --dropout 0.2
```

### 3. Hyperparameter Tuning 실행

```bash
python3 train_lstm_markers_test.py --tune --n_trials 100
```

## Arguments 상세 설명

### 출력 설정

| Argument | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `--out_dir` | str | `./runs/lstm_YYYYMMDD-HHMMSS` | 학습 결과가 저장될 디렉토리 경로 |

### 데이터 윈도우 설정

| Argument | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `--window` | int | 60 | 입력 시계열 윈도우 크기 (과거 몇 개 시점을 볼 것인가) |
| `--horizon` | int | 1 | 예측 지평선 (미래 몇 시점을 예측할 것인가) |
| `--stride` | int | 1 | 슬라이딩 윈도우 스트라이드 (데이터 샘플링 간격) |

**예시:**
- `--window 60 --horizon 1`: 과거 60 시점을 보고 1 시점 후 예측
- `--window 100 --horizon 5`: 과거 100 시점을 보고 5 시점 후 예측

### 데이터 분할 비율

| Argument | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `--train_ratio` | float | 0.8 | 전체 데이터 중 학습 데이터 비율 |
| `--val_ratio` | float | 0.1 | 전체 데이터 중 검증 데이터 비율 |

나머지는 자동으로 테스트 데이터로 할당됩니다.  
**예시:** train 80% → val 10% → test 10%

### 학습 하이퍼파라미터

| Argument | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `--epochs` | int | 30 | 학습 에포크 수 |
| `--batch_size` | int | 256 | 배치 크기 (GPU 메모리에 따라 조정) |
| `--lr` | float | 0.001 | Learning rate (학습률) |

**권장 범위:**
- `--epochs`: 20~100 (데이터 크기에 따라)
- `--batch_size`: 64, 128, 256, 512 (2의 거듭제곱 추천)
- `--lr`: 0.0001~0.01 (너무 크면 발산, 너무 작으면 느림)

### LSTM 모델 구조

| Argument | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `--hidden_size` | int | 128 | LSTM 은닉층 크기 |
| `--num_layers` | int | 2 | LSTM 레이어 개수 |
| `--dropout` | float | 0.1 | Dropout 비율 (과적합 방지) |

**권장 범위:**
- `--hidden_size`: 64, 128, 256, 512 (클수록 표현력 증가, 과적합 위험)
- `--num_layers`: 1~4 (깊을수록 복잡한 패턴 학습 가능)
- `--dropout`: 0.0~0.5 (높을수록 과적합 방지 효과 증가)

### 입력 데이터 설정

| Argument | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `--input_col` | str | `data[1]` | 입력으로 사용할 컬럼 이름 (CSV 파일 내) |

**사용 가능한 컬럼:**
- `data[0]`: target_pressure
- `data[1]`: current_pressure (기본값)
- `data[2]`: filtered_pressure
- `data[3]`: valve

### Hyperparameter Tuning

| Argument | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `--tune` | flag | False | Hyperparameter tuning 활성화 (Optuna 사용) |
| `--n_trials` | int | 50 | Tuning trial 횟수 |

**Tuning 범위:**
- `window`: 10~100 (step=10)
- `horizon`: 1~10
- `stride`: 1~5
- `batch_size`: [64, 128, 256, 512]
- `epochs`: 20~50
- `lr`: 0.0001~0.01 (log uniform)
- `weight_decay`: 0.00001~0.001 (log uniform)
- `hidden_size`: [64, 128, 256, 512]
- `num_layers`: 1~4
- `dropout`: 0.0~0.5
- `grad_clip`: 0.5~2.0

## 사용 예시

### 예시 1: 빠른 실험 (적은 epoch)

```bash
python3 train_lstm_markers_test.py \
  --epochs 10 \
  --batch_size 512 \
  --out_dir ./runs/quick_test
```

### 예시 2: 긴 시퀀스 학습

```bash
python3 train_lstm_markers_test.py \
  --window 120 \
  --horizon 5 \
  --stride 2 \
  --epochs 60
```

### 예시 3: 큰 모델 학습

```bash
python3 train_lstm_markers_test.py \
  --hidden_size 512 \
  --num_layers 4 \
  --dropout 0.3 \
  --batch_size 128 \
  --epochs 80
```

### 예시 4: Hyperparameter Tuning (자동 최적화)

```bash
python3 train_lstm_markers_test.py \
  --tune \
  --n_trials 100 \
  --out_dir ./runs/tuning_results
```

### 예시 5: 다른 입력 컬럼 사용

```bash
# Filtered pressure를 입력으로 사용
python3 train_lstm_markers_test.py --input_col "data[2]"

# Target pressure를 입력으로 사용
python3 train_lstm_markers_test.py --input_col "data[0]"
```

### 예시 6: 데이터 분할 비율 변경

```bash
# Train 70%, Val 20%, Test 10%
python3 train_lstm_markers_test.py \
  --train_ratio 0.7 \
  --val_ratio 0.2
```

## 출력 파일 설명

학습 완료 후 `--out_dir` 폴더에 다음 파일들이 생성됩니다:

### 모델 관련
- **`best.pt`**: 검증 손실이 가장 낮았던 모델 체크포인트
- **`last.pt`**: 마지막 에포크의 모델 체크포인트
- **`config.json`**: 학습에 사용된 전체 설정값
- **`norm_stats.npz`**: 정규화 통계값 (평균, 표준편차)

### 평가 결과
- **`test_metrics.json`**: 테스트 세트 상세 평가 지표
  - 전체 MAE, MSE, RMSE
  - 마커별 MAE, RMSE
  - 축별(X/Y/Z) MAE
  
### 시각화
- **`training_history.png`**: 학습/검증/테스트 손실 그래프
- **`test_predictions.png`**: 예측값 vs 실제값 시계열 비교 (마커별)
- **`test_errors.png`**: 마커별 MAE + 에러 분포 히스토그램
- **`test_scatter.png`**: 예측값 vs 실제값 산점도

### Tuning 결과 (--tune 사용 시)
- **`tuning_results.json`**: 최적 hyperparameter 및 성능
- **`tuning_history.png`**: Optimization 진행 과정
- **`param_importances.png`**: 파라미터 중요도 분석

## Tips & Tricks

### 1. GPU 메모리 부족 시

```bash
# 배치 크기 줄이기
python3 train_lstm_markers_test.py --batch_size 64

# 모델 크기 줄이기
python3 train_lstm_markers_test.py --hidden_size 64 --num_layers 2
```

### 2. 과적합(Overfitting) 발생 시

```bash
# Dropout 증가
python3 train_lstm_markers_test.py --dropout 0.3

# 또는 작은 모델 사용
python3 train_lstm_markers_test.py --hidden_size 64 --num_layers 1
```

### 3. 학습이 너무 느릴 때

```bash
# Learning rate 증가
python3 train_lstm_markers_test.py --lr 0.005

# 배치 크기 증가 (GPU 메모리가 충분하다면)
python3 train_lstm_markers_test.py --batch_size 512
```

### 4. 더 정확한 예측이 필요할 때

```bash
# 긴 입력 시퀀스 사용
python3 train_lstm_markers_test.py --window 150

# 큰 모델 사용
python3 train_lstm_markers_test.py --hidden_size 256 --num_layers 3

# 더 많은 에포크
python3 train_lstm_markers_test.py --epochs 100
```

## 학습 결과 해석

### 손실(Loss) 지표
- **Train Loss < Val Loss**: 정상 (모델이 학습 중)
- **Train Loss ≈ Val Loss**: 좋은 상태 (과적합 없음)
- **Train Loss << Val Loss**: 과적합 (dropout 증가 또는 모델 축소 필요)

### MAE (Mean Absolute Error)
- 예측값과 실제값의 평균 절대 오차
- **낮을수록 좋음**
- 정규화된 값 기준 (원본 단위로 변환 필요 시 `norm_stats.npz` 사용)

### RMSE (Root Mean Square Error)
- MAE보다 큰 오차에 더 민감
- 이상치(outlier)에 대한 평가
- **낮을수록 좋음**

## 문제 해결

### ImportError: optuna
```bash
pip install optuna plotly kaleido
```

### CUDA out of memory
- `--batch_size`를 64 또는 32로 줄이기
- `--hidden_size`를 128 또는 64로 줄이기

### ValueError: Missing columns
- 입력 CSV 파일에 필요한 컬럼이 있는지 확인
- `process_and_merge_data.py`를 먼저 실행했는지 확인

## 데이터 경로

현재 코드는 다음 경로의 데이터를 사용합니다:
```
/home/min/project_SORO/data/rosbag2_2026_03_03-16_14_02/new_merged_data.csv
/home/min/project_SORO/data/rosbag2_2026_03_03-16_24_27/new_merged_data.csv
```

다른 데이터를 사용하려면 코드 내 경로를 직접 수정하거나, 향후 argument로 추가하세요.

## 참고사항

- 모든 마커 위치는 **marker_2를 중심점으로 한 상대 위치**로 예측됩니다
- 입력 데이터와 출력 데이터는 자동으로 정규화되며, 통계값은 `norm_stats.npz`에 저장됩니다
- 학습 중 자동으로 best model이 저장되며, 테스트는 best model로 수행됩니다
- Hyperparameter tuning 시 early stopping이 적용되어 효율적으로 탐색합니다

## 문의

추가 질문이나 버그 리포트는 프로젝트 관리자에게 문의하세요.
