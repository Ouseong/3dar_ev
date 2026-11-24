# 1개 스피커 정상 vs 고장 — 직접 픽셀 비교 (2026-11-23)

사용자 지시로 CLAUDE.md 3번(1대 200개 학습 제외)과 별개로 1개 스피커 녹음 200개를 분석함. 학습은 아직 안 함.

## 데이터
- `prepare_single.py` → `runs/ae_paper_single_20261123/manifest.json`, `splits/seed{42,43,44}.json`
  - 정상 60 (위치 3 × RPM 2 × 10), 고장 140 (앞/뒤: Imbalance 20, PWM 20, Modulation 10(A만), Whine 10(B만); 오른쪽: Wind 20)
  - 분할: 정상 36/12/12 (학습/검증/시험), 고장 70/70 (검증/시험). 원본 음원 100개, 최대 3회 재사용 (closed-source).
- 통째 녹음 캐시 `build_cache_single.py` (STFT 0–12 kHz 1536칸 × 384열, `runs/ae_paper_single_20261123/cache/`)

## 방법 (`compare_pixels.py [--feature mel|stft6k]`)
- 같은 위치·같은 RPM 대역의 정상 10개 평균을 기준으로, 각 녹음의 dB 차이. 정상은 leave-one-out.
- 픽셀 RMS, 시간 평균 스펙트럼 RMS, 정상 범위(3σ) 밖 픽셀 비율.

## 결과: 픽셀 RMS (dB), 괄호 = 정상 대비 배율
| | STFT 0–12 kHz 1536 (7.8 Hz) | mel 64 (0–12 kHz) | STFT 0–6 kHz 1536 (3.9 Hz) |
|---|---|---|---|
| 정상 vs 정상 | 6.26 | 4.54 | 6.78 |
| Imbalance | 6.56 (1.05) | 4.89 (1.08) | 6.91 (1.02) |
| Modulation | 7.51 (1.20) | 5.89 (1.30) | 7.90 (1.17) |
| Whine | 6.82 (1.09) | 5.50 (1.21) | 7.53 (1.11) |
| PWM | 6.48 (1.04) | 5.13 (1.13) | 6.91 (1.02) |
| Wind | 7.24 (1.16) | 5.43 (1.20) | 7.35 (1.08) |

- 그림: `figures/single_pixel_compare.png`, `_mel.png`, `_stft6k.png`. 수치: `runs/ae_paper_single_20261123/pixel_compare*.json`
- 칸이 촘촘할수록 칸별 잡음이 커져 대비가 줄어듦. mel(여러 칸 평균)이 대비가 가장 큼. 0–6 kHz 3.9 Hz는 개선 없음.
- 논문 설계대로 보이는 고장 무늬: Whine(1.7–2.1 kHz order 16 선, 뚜렷), PWM(5 kHz 선 +3–4 dB, STFT에서만). 나머지(Imbalance 80–270 Hz, Modulation 측대역, Wind 3.75–5.75 kHz 대역)는 안 보임.
- Wind는 오히려 넓은 대역이 작게 녹음됨. Modulation/Imbalance는 넓은 대역이 커짐 → 설계 성분이 아닌 차이로 보임.

## 혼란 요인 (confound)
- 녹음 순서가 클래스별 블록: 정상 전부 19:52–20:06, 이후 고장(Imbalance 앞 20:06 … Wind 20:26–20:29). 시간에 따른 주변 소음·장비 변화가 라벨과 겹침.
- 312 Hz 고정 톤: 정상 녹음은 주변 대비 약 2 dB, 20:06 이후 고장 녹음은 약 4–6 dB. 모든 고장 클래스에 공통 → 설계된 고장 성분이 아님. 원본 음원 차이인지 녹음 중 환경 변화인지 미확인(원본 음원 WAV 없음).
- 시작 직후(약 0.15 s) 1 kHz·6 kHz 덩어리(Wind/PWM/Whine 평균에서 큼), 정상 일부의 10 kHz 톤.
- → 이 데이터로 정상/고장을 잘 구분해도 고장음이 아니라 녹음 시점을 구분한 것일 수 있음.
