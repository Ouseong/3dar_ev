# 사전학습 특징(BEATs, YAMNet)으로 지도·비지도 정상/고장 판정 (2026-11-23)

- 사용자 요청으로 CLAUDE.md 4번(YAMNet·BEATs 금지)과 별개 실험으로 진행. 기존 코드·결과 수정 없음.
- 모델: BEATs `BEATs_iter3_plus_AS2M.pt`(Microsoft, 768차원; HF 미러 lpepino/beats_ckpts, sha256 d43cbfad…, weights_only 로드), YAMNet `yamnet.h5`(Google 공식, 1024차원). 펭귄 노트북처럼 16 kHz, 프레임 평균, YAMNet은 0.975 s 미만 0 패딩. 8 kHz 이상 정보는 잘림.
- 환경: torchaudio가 torch 2.14.1/aarch64용으로 없어 `third_party/torchaudio_shim`(kaldi.py만, torchaudio 2.11.0 wheel에서 복사). YAMNet은 별도 venv `/home/kos/Desktop/3DAR_EV/.venv_tf`(TensorFlow 2.18 CPU).
- 코드: `extract_segments.py` → `embed_beats.py` / `embed_yamnet.py` → `evaluate.py`. 결과 `runs/pretrained_emb_20261123/summary.json`.
- 지도: 엄격 분할(원본 음원 안 겹침) seed 42/43/44, 표준화 + 로지스틱 회귀(클래스 균형, C는 검증 AUC로 선택). 비지도: 고장 0곳 녹음 조각만 기준(5-fold, 원본 음원 겹치지 않는 기준 모델로만 점수), PCA 32 + Ledoit-Wolf Mahalanobis.

| | BEATs 지도 | YAMNet 지도 | 둘 합침 지도 | CNN 지도(참고) | BEATs 비지도 | YAMNet 비지도 |
|---|---|---|---|---|---|---|
| AUC 전체 | **0.958** | 0.835 | 0.940 | 0.949 | 0.711 | 0.652 |
| 정확도 | 0.901 | 0.784 | 0.877 | 0.875 | — | — |
| AUC 어려운 정상만 | 0.861 | 0.689 | 0.825 | 0.829 | 0.448 | 0.534 |
| 1·2곳 녹음 안 조각 AUC | 0.784 | 0.695 | 0.756 | 0.731 | 0.517 | 0.583 |
| 전부 정상 vs 전부 고장 녹음 | 0.992 | 0.878 | 0.979 | — | 0.925 | 0.726 |
| 깨끗한 정상 vs 고장 녹음 속 정상 조각 | 0.857 | 0.720 | 0.826 | — | **0.950** | 0.745 |
| 1곳 녹음 고장 조각 1등 (무작위 .33) | 0.48 | **0.59** | 0.45 | 0.44 | 0.33 | 0.44 |
| 재현율 Imb/Mod/Whine/PWM/Wind | .73/.95/.89/.92/.86 | .77/.86/.93/.87/.52 | .68/.95/.85/.85/.86 | .87/.97/.80/.90/.88 | — | — |

- BEATs는 CNN과 비슷하거나 조금 나음. 같은 한계: 고장 녹음 속 정상 조각도 고장으로 봄(비지도 0.95 → 파일 묶음 차이를 강하게 잡음).
- YAMNet은 전체 성능은 낮지만 파일 묶음 차이에 덜 민감(0.72)하고 고장 조각 1등 0.59로 지금까지 중 가장 높음. 단 시험 고장 1곳 녹음이 seed당 10개 안팎이라 불확실.
- 비지도는 방향 구분 못 함(1등 0.33–0.44).
