# 3DAR_EV AI 실행 묶음

사용자의 실제 녹음 구조에 맞춰 펭귄 AI 코드를 바탕으로 조정한 **지도학습 기준 모델**이다. Claude Code는 먼저 `CLAUDE.md`와 `HANDOFF_FOR_CLAUDE.md`를 읽는다. 데이터와 모델 학습 결과는 ZIP에 들어 있지 않다.

## 파일 배치

```text
작업폴더/
├── EV_3DAR_AI/             ← ZIP을 풀면 생기는 코드 폴더
│   ├── CLAUDE.md
│   ├── configs/ev.json
│   ├── ev3dar/
│   ├── prepared/           ← 실제 자료로 미리 만든 목록·분할·해시 (음원 없음)
│   └── validation/         ← 학습 없이 수행한 점검 결과
└── 3DAR_EV/                ← 사용자가 따로 전달하는 측정 자료
    ├── 측정_목록.csv
    └── 측정_데이터/…
```

별도 전달하는 `3DAR_EV`는 **WAV와 두 CSV 및 측정_목록.csv가 있는 전체 폴더**여야 한다. 원본 KU Leuven 재생 음원은 이번 모델 실행에 필요 없다. `Source_File`은 원본 재사용 감사용 문자열로만 쓰고 파일을 열지 않는다.

## Spark에서 실행

노트북에서는 학습하지 않는다. DGX Spark의 **CUDA 사용이 가능한 PyTorch 환경**에서 다음 명령을 실행한다. GPU가 없으면 기본 학습은 오류로 멈추며 CPU로 자동 전환하지 않는다.

```bash
cd /경로/EV_3DAR_AI
python scripts/verify_environment.py
# 기본 라이브러리가 없을 때만 설치. 기존 GPU PyTorch를 교체하지 않는다.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/verify_dataset.py --data-root ../3DAR_EV --prepared prepared --forward

# 먼저 seed42·1 epoch로 실제 GPU 학습/저장/평가 연결 확인
bash smoke_spark.sh /절대경로/3DAR_EV

# 위 점검이 통과하면 별도 출력 폴더에서 본 학습: seeds42,43,44
bash run_spark.sh /절대경로/3DAR_EV
```

PyTorch 설치 명령은 ARM/CUDA 환경마다 달라 자동 포함하지 않았다. GPU 환경 점검이 실패하면 Spark에 맞는 기존 환경/컨테이너를 먼저 사용한다. 위 smoke는 **실제 학습**이므로 Spark에서만 실행한다. Windows 경로가 코드에 고정되어 있지 않다.

## 무엇을 학습하는가

| 항목 | 현재 구현 |
|---|---|
| 자료 | 3대 스피커 녹음 1,200개만 사용; 1대 녹음 200개 제외 |
| 분할 | 고장 위치 수 × RPM 구간을 유지해 녹음별 960/120/120; seeds42/43/44 |
| SSL: 음원 위치 | 각도 CSV로 정렬한 Log-Mel → 원형 1D CNN → 전체 음원 수 + 360칸 방향 점수 |
| ID: 소리 종류 | 각 방향의 수치 스펙트로그램 → 2D CNN → 벡터 → MLP/latent → 6종 분류 |
| 6종 | Healthy / Imbalance / Modulation / Whine / Wind / PWM |
| 결합 | SSL 방향 점수와 해당 영역의 ID 확률을 곱한 360×6 공간 음향 맵 |
| RPM | 기본 입력 제외; 선택 시 ID 벡터에만 concat; 추정값이지 실측 회전수 아님 |
| 제외 | YAMNet, BEATs, cross-attention, SupCon, temperature/lambda 탐색 |

`decoder`는 **종류 분류 층**이다. 오토인코더 복원 학습, 정상만 학습하는 이상 탐지, 물리적 손상 부품 판별을 구현했다고 부르면 안 된다. 현재 목표는 녹음에 붙은 소리 종류와 해당 위치의 정상/고장을 지도학습으로 구분하는 것이다.

## 각도·소리 처리

- 음성 PNG를 입력하지 않고 mono48kHz WAV에서 수치 배열을 계산한다.
- Hann FFT2048, hop480(10ms), 64Mel, 0–12kHz. 기존 표시 PNG의 6kHz보다 넓게 보도록 정한 초기 설정이며, 실제 PWM 특징 대역을 확인하고 필요하면 변경한다.
- SSL: 실제 각도 로그에 맞춰 0–359°로 정렬. 각 1° 지점의 중앙 특징과 주변 **1초 문맥의 평균 파워 특징**을 함께 입력한다. 1° 간격은 입력 좌표이며 물리적 1° 분해능을 보장하지 않는다.
- ID: 앞135–225°, 오른쪽225–315°, 뒤315–405°. 앞·오른쪽은 약1초, 뒤는 실제315–360°와0–45° 두 구간을 **각각 인코딩한 뒤 시간 가중 평균**한다. 두 WAV 끝을 연속 음원처럼 이어 붙이지 않는다.
- 크롭마다 독립 음량 정규화하지 않는다. 학습 fold에서만 계산한 주파수별 평균·표준편차를 검증/평가에도 적용한다. 숫자 세기는 dB SPL 보정값이 아니다.
- 앞180°, 뒤0°, 오른쪽270°는 현재 설정 가정이다. 시계/반시계와 실제 설치 방향을 확인하기 전 절대 위치 정확도라고 주장하지 않는다.

## 반드시 알아야 할 평가 한계

1. **현재 1,200개 모두 같은 위치에서 스피커3개를 재생했다.** 개수3과 세 방향이라는 SSL 정답이 상수다. 입력을 안 보고 항상3과 고정 방향을 내도 개수 정확도100%, 기준 배치 각도오차0°다. 이를 넘는 일반적 위치 추정 검증은 현재 자료로 할 수 없다. 코드는 이 상수 기준선도 함께 기록한다.
2. 재생 원본879개 중726개가 여러 녹음에 재사용되었다. 원본 공유 연결성분 최대930개에 모든 고장 혼합 녹음이 들어 있어, 전부 사용하면서 엄격한 원본 비중복8:1:1 분할은 불가능하다. 현재 결과는 **기지 음원의 서로 다른 녹음에 대한 평가**다. 미지 음원 일반화 성능이 아니다.
3. 음원 종류 라벨이 실제로 어느 부품이 손상되었는지 증명하지 않는다. 실제 차량 고장 진단 성능이라고 표현하지 않는다.
4. 1초 SSL 문맥·1° stride와 숫자 방위각은 구현상 가정으로 명시했다. 바꿀 때 별도 출력 폴더를 사용하고 전처리부터 다시 수행한다.

## 결과 파일

```text
runs/baseline/
├── manifest.jsonl / data_audit.json / config.resolved.json
├── splits/seed42.json …
├── cache/                 # 전처리 특징만, 전송 ZIP에는 없음
├── seed42/                # seed43/44도 같은 구조
│   ├── normalization.json
│   ├── ssl_best.pt / id_best.pt
│   ├── ssl_history.json / id_history.json
│   ├── test_metrics.json
│   ├── test_predictions.csv
│   ├── type_confusion_matrix.csv
│   └── spatial_maps/*.npz
└── three_seed_summary.json
```

예측 CSV는 녹음마다 앞·뒤·오른쪽 **3행**이며 종류 확률, 정상/고장 확률·예측, 정답, 추정 각도, 음원 수를 기록한다. 지표는 실제 Top1 정확도·macroF1·정상/고장 F1·balanced accuracy·ROC AUC/AP와 방향 점수다. AUC는 임계값을 바꿨을 때 정상보다 고장 점수가 높은지 평가하며, 정확도와 같은 뜻이 아니다. 고정 위치별 학습 클래스 비율만 사용하는 기준선도 비교한다.

전처리 캐시는 재실행 시 일치하는 파일을 재사용한다. **중단된 학습의 optimizer/epoch 자동 복원은 구현하지 않았다.** `train`을 재실행하면 해당 모델을 처음부터 다시 학습한다. 출력 폴더의 설정을 바꾸어 기존 결과와 섞지 않는다. 학습한 뒤 목록/분할/설정을 바꾸면 체크포인트 평가가 거부된다.

## 수동 단계 실행

```bash
python -m ev3dar prepare --data-root ../3DAR_EV --out runs/baseline
python -m ev3dar features --data-root ../3DAR_EV --out runs/baseline
python -m ev3dar train --out runs/baseline --seed 42 --task ssl
python -m ev3dar train --out runs/baseline --seed 42 --task id
python -m ev3dar evaluate --out runs/baseline --seed 42
# seed43/44도 학습/평가 후
python -m ev3dar summarize --out runs/baseline
```

현재 ZIP은 실행 가능한 기준 코드이지 학습이 끝난 모델이 아니다. 로컬 검증 범위는 `validation/`을 확인한다.
