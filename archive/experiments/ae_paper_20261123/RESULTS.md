# 논문 AE 구조 적용 — 정상만 학습, 구간별 고장 탐지 (2026-11-23)

## 출처
- Kunte et al., "A Domain Knowledge Informed Approach for Anomaly Detection of Electric Vehicle Interior Sounds" (KU Leuven, SSRN preprint) Fig. 10 / Table 5.
- 인코더 블록 ×n: Conv2d(f, 3×3, s1, p1) → BatchNorm2d → ReLU → MaxPool2d(2,2)
- 디코더 블록 ×(n−1): ConvTranspose2d(f) → Conv2d(f) → BatchNorm2d → ReLU, 출력: ConvTranspose2d(f) → Conv2d → Sigmoid. Dense 층 없음.
- Adam lr 1e-4, batch 8, 최대 100 epoch, MSE. 입력은 0–120 dB clip 후 [0,1] (데이터 전체 기준).
- 논문은 n=1–7 × f=4/8/16/32 (28개)를 proxy anomaly(합성 가짜 고장)로 선택 → n=6, f=8.

## 우리 데이터에 맞춘 점 (논문에 명시 없음)
- ConvTranspose2d kernel 2 stride 2 (MaxPool 되돌림), 마지막 Conv2d 출력 1채널.
- dBFS 특징이라 clip(−120, 0)/120 + 1. 시간축 100 → 128 padding (mask로 loss 제외).
- proxy anomaly 자료가 없어 모델 선택은 하지 않고 논문 선택값 n=6, f=8 고정.
- 나머지(ROI, 분할, 정상만 학습, masked MSE, 검증 임계값, 지표)는 `experiments/ae_20261119/train_ae.py`와 동일 (코드 재사용).
- 조기 종료: 검증 정상 복원 오차, patience 15.

## 실행
- 코드 `train_ae_paper.py --blocks 6 --filters 8` (seed 42/43/44), 출력 `runs/ae_paper_20261123/n6_f8/`, 로그 `runs/ae_paper_20261123/n6_f8.log`.
- 구조도: `figures/ae_paper_architecture.png`(입력 포함), `figures/ae_paper_hourglass.png`(AE만).
- 파라미터 7,753개. 다른 프로젝트(Isaac Sim)가 GPU를 같이 쓰는 중에 실행, 우리 학습은 GpuThrottle 50%.

## 시험 결과 (구간 360개, 3 seed 평균 ± 표준편차)
| 지표 | 논문 AE n6 f8 | 11/19 ConvAE |
|---|---|---|
| ROC-AUC | 0.706 ± 0.043 | 0.797 ± 0.005 |
| 정확도 | 0.665 ± 0.031 | 0.750 |
| 고장 재현율 | 0.715 ± 0.087 | 0.880 |
| 정상 특이도 | 0.615 ± 0.077 | 0.620 |
| F1 | 0.679 ± 0.042 | 0.778 |

- seed별 AUC: 0.705 / 0.655 / 0.759. best epoch 97 / 99 / 97 (100 epoch 끝까지 검증 오차가 계속 감소 → lr 1e-4·100 epoch로는 수렴 전).
- 정상 오탐률 (고장 0 / 1 / 2곳 녹음): seed42 0.20/0.40/0.37, seed43 0.33/0.60/0.70, seed44 0.20/0.57/0.53.

## 해석·한계
- 논문 구조를 그대로 옮긴 결과는 기존 ConvAE보다 AUC가 약 0.09 낮고 seed 편차가 큼.
- 가능 원인(가설): 입력이 논문(1536×384 order spectrogram)보다 훨씬 작은 64×128이라 n=6이면 latent가 16개 값뿐임; 학습이 100 epoch 안에 수렴하지 않음; 논문 모델 선택(proxy anomaly)을 재현하지 않음.
- closed-source holdout, 임계값은 검증 고장 라벨로 정함 — 11/19 AE와 같은 한계.

## 추가: 선형 STFT 1536칸 입력 (같은 날)
- mel 64칸이 고장 선(Whine/PWM/Modulation)을 뭉갠다는 판단으로 논문처럼 주파수 1536칸 입력을 만듦.
- 캐시 `build_cache_stft.py` → `runs/ae_paper_20261123/stft_cache/` : n_fft 6144(7.8 Hz, 128 ms Hann), hop 480, 0–11,992 Hz 1536칸, ROI·각도·100 frame 규칙은 기존과 동일. mel 압축 없음.
- 실행 `train_ae_paper.py --feature stft --blocks 6 --filters 8` → `runs/ae_paper_20261123/stft_n6_f8/`. 잠재 8×24×2=384.

| 지표 | STFT 1536 n6 f8 | mel 64 n6 f8 |
|---|---|---|
| ROC-AUC | 0.560 ± 0.007 | 0.706 ± 0.043 |
| 정확도 | 0.575 ± 0.012 | 0.665 |
| 재현율 | 0.367 ± 0.079 | 0.715 |
| 특이도 | 0.783 ± 0.061 | 0.615 |

- seed별 AUC 0.551 / 0.566 / 0.565, best epoch 100 / 100 / 99 (여전히 미수렴).
- 거의 구분 못 함. 추정 원인: 입력 24배인데 필터 8개(7.7k 파라미터), 가는 고장 선이 전체 MSE 평균에 묻힘, 수렴 전. 검증하지 않은 가설.
