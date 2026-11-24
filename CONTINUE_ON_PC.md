# 개인 컴퓨터(VS Code)에서 이어서 작업하기

Spark(DGX)에서 만든 코드를 개인 컴퓨터의 VS Code로 받아 구조를 살펴보고 정리하기 위한 안내입니다.
(이 문서에서 "주피터 노트북"은 `.ipynb` 파일, "개인 컴퓨터"는 사용자의 PC를 뜻합니다.)

## 1. 코드 받기
```bash
git clone https://github.com/Ouseong/3dar_ev.git
cd 3dar_ev
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# PyTorch 먼저 설치 (GPU가 있으면 CUDA 버전): https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

## 2. 데이터 준비 (저장소에는 데이터가 없음)
둘 중 하나를 고른다.

**A. 측정 데이터 전체 (특징을 새로 뽑을 때)**
- Google Drive 폴더 `3DAR Electric Car/3DAR_EV`(측정_목록.csv, 측정_데이터/)를 PC로 받는다. 약 5,600개 파일.
- 노트북 `CONFIG['data_root']`를 `측정_목록.csv`가 있는 폴더로 바꾼다.

**B. 특징 캐시만 (빠름, 코드 정리용으로 충분)**
- Spark에서 아래 파일을 복사한다(약 30 MB). 공개 저장소에는 올리지 않는다.
  `/home/kos/Desktop/3DAR_EV/EV_3DAR_AI/runs/nb_faultbandnet_20261124/features_81b146e716ad05ea.npz`
  ```bash
  mkdir -p runs/nb_faultbandnet_20261124
  scp <spark-user>@<spark-host>:/home/kos/Desktop/3DAR_EV/EV_3DAR_AI/runs/nb_faultbandnet_20261124/features_81b146e716ad05ea.npz runs/nb_faultbandnet_20261124/
  ```
- 이때도 CSV(측정_목록.csv, 녹음별 실험정보·각도 CSV)는 필요하다. Drive에서 CSV만 받아도 된다.
- 특징 설정(대역, n_fft 등)을 바꾸면 캐시 이름(hash)이 달라져 WAV가 필요하다.

## 3. 실행
- VS Code(Jupyter 확장)로 `3DAR_EV_FaultBandNet.ipynb`를 열고 셀 1 → 2 → 3 → 4 순서로 실행한다.
- 주 노트북은 3×3을 30회 반복 학습한다(Spark GPU에서 약 30분). 서브 노트북 `sub/3DAR_EV_FaultBandNet_3x1.ipynb`는 3×1이다.
- 학습을 건너뛰고 저장된 결과로 평가·그림만 보려면 `CONFIG['train'] = False`(결과 파일이 `runs/`에 있어야 함).
- GPU가 없으면 `CONFIG['device']`를 바꿔야 한다. 모델이 작아(14.5만 개 파라미터) CPU로도 돌지만 오래 걸린다.
- 결과는 `runs/nb_faultbandnet_strict_20261124/`에 저장된다(저장소에는 안 올라감). 서브 노트북도 `../runs/`의 같은 폴더를 쓴다.

## 4. Claude Code에 처음 보낼 프롬프트
VS Code에서 저장소 폴더를 열고, 터미널(또는 Claude Code 확장)에서 `claude`를 실행해 아래를 붙여 넣는다. `CLAUDE.md`(작업 지침)는 자동으로 읽힌다.

```text
이 저장소는 3DAR(회전 메타구조 마이크)로 전기차 고장음의 위치와 종류를 찾는 실험이야.
Spark(DGX)에서 하던 작업을 지금 이 개인 컴퓨터의 VS Code에서 이어서 하려고 해.

먼저 CLAUDE.md, README.md, CONTINUE_ON_PC.md를 읽고,
주피터 노트북 3DAR_EV_FaultBandNet.ipynb (3×3, 주 모델) 셀 1–4를 실제로 끝까지 읽어서 구조를 파악해 줘.
archive/experiments/fault_net_noimbmod_20261124/RESULTS.md 도 읽어 줘.

목표는 3×3 FaultBandNet 코드 구조를 뜯어보고 깔끔하게 정리하는 거야.
1) 코드 구조를 짧게 요약하고(입력·모델·분할·평가, 함수별 역할과 호출 순서),
2) 정리할 점을 목록으로 제안해 줘(중복 코드, 3×1 분리 여부, 이름, 주석, 셀 나누기 등).
결과가 바뀌는 수정과 단순 정리는 구분해서 제안하고, 내가 고르기 전에는 고치지 마.

데이터 경로는 <여기에 data_root 경로>, 특징 캐시는 runs/nb_faultbandnet_20261124/ 에 있어.
GPU는 <있음/없음>.
```
