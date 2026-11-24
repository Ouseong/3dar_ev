"""Write 3DAR_EV_FaultBandNet.ipynb — strict version (2026-11-24, user request "펭귄처럼 하나로 정리").
Same penguin layout and the same feature/model code as experiments/fault_net_20261124/make_notebook.py, changed to:
  * recordings with Imbalance/Modulation removed (664 of 1,200)
  * repeated source-disjoint splits (no original file of ANY speaker shared by train and test), 30 repeats
  * main notebook = 3x3 (position x fault type); sub notebook = 3x1 (position: faulty or not), same code otherwise
Usage: python make_notebook.py  -> EV_3DAR_AI/3DAR_EV_FaultBandNet.ipynb (3x3), EV_3DAR_AI/sub/3DAR_EV_FaultBandNet_3x1.ipynb
"""
from pathlib import Path
import json, sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fault_net_20261124'))
import make_notebook as OLD


def sub(s, a, b):
    assert s.count(a) == 1, a; return s.replace(a, b)


MD = """# 3DAR_EV FaultBandNet — 고장 대역 × 마이크 각도 지도로 고장 위치(·종류) 판정

전기차 고장음을 3DAR(회전 메타구조 마이크)로 녹음한 뒤, **고장이 나타나는 대역만** 각도별로 뽑아 신경망에 넣는다.

- 입력: 녹음 1개(한 바퀴) → 지도 3장 (마이크 각도 36칸 × 주파수), 모두 녹음 안 상대값
  - Wind 3.3–6.2 kHz (36×58) · PWM 4.8–5.2 kHz (36×68) · Whine order 16 ± 150 Hz (36×51)
  - 각도는 360° 전부, 주파수는 0–24 kHz 중 약 13 %만 사용
- 모델: 지도별 2D CNN → 각도 방향 1D CNN(원형) → FC → **3×3 = 위치 3(앞/뒤/오른쪽) × 고장 3(Whine/PWM/Wind) 확률**
  (위치별 고장 여부만 내는 3×1 변형은 서브 노트북 `sub/3DAR_EV_FaultBandNet_3x1.ipynb`)
- 데이터: 3스피커 녹음 **1,200개 전부**. Imbalance(1·2차, 0.3 kHz 미만)·Modulation(약 8차, 1.1 kHz 미만)은 지도 대역 밖이라
  **정상(0)으로 학습·평가**한다. 즉 정답은 "그 위치에 Whine/PWM/Wind가 있나"이고, 고장 개수도 이 세 고장만 센다.
- 평가: **원본 음원 비중복 반복 분할 30회, 고장 개수(0/1/2/3곳)별 같은 비율**.
  시험 = 고장 개수마다 약 5.3 %, 학습 = 시험 원본 파일이 하나도 없는 녹음에서 고장 개수마다 같은 비율만 남김,
  검증 = 학습 풀에서 고장 개수마다 15 %. 어떤 스피커의 원본 파일도 학습과 시험에 겹치지 않는다. 녹음별로 시험된 회차의 확률을 평균.
  (이전 버전의 5-fold는 오른쪽 스피커만 묶어서 시험 녹음의 98 %가 앞·뒤 원본을 학습과 공유했음)

셀 구성(펭귄 노트북과 같은 순서): **셀 1** 설정·데이터·특징·모델·학습 → **셀 2** 평가·요약 → **셀 3–4** 그림.
데이터(측정 WAV)는 저장소에 포함하지 않는다. `CONFIG['data_root']`만 바꾸면 된다.
"""

C1 = OLD.CELL1
C1 = sub(C1, "    'out_dir': 'runs/nb_faultbandnet_20261124',                            # 이 노트북의 모든 결과\n",
         "    'out_dir': '@PRE@runs/nb_faultbandnet_all_20261124',                       # 이 노트북의 모든 결과\n"
         "    'feature_cache_dir': '@PRE@runs/nb_faultbandnet_20261124',                  # 특징 캐시(녹음 1,200개) 재사용\n")
C1 = sub(C1, "    'folds': 5, 'validation_fraction': 0.15, 'seed_base': 100,\n",
         "    'exclude_classes': [],                                                          # 뺄 녹음 없음 (1,200개 전부)\n"
         "    'healthy_like': ['Imbalance', 'Modulation'],                                    # 지도 대역 밖 → 정상(0)으로 학습·평가\n"
         "    'output_mode': '@MODE@',                                                         # 3x3: 위치×고장종류, 3x1: 위치×고장여부\n"
         "    'repeats': 30, 'test_rate': 0.053, 'split_seed': 0,                             # 원본 음원 비중복 반복 분할, 고장 개수별 같은 비율\n"
         "    'validation_fraction': 0.15, 'seed_base': 1000,\n")
C1 = sub(C1, "from sklearn.model_selection import GroupKFold, GroupShuffleSplit\n", "")
C1 = sub(C1, "    cache = OUT / f'features_{key}.npz'", "    cache = Path(CONFIG['feature_cache_dir']) / f'features_{key}.npz'; cache.parent.mkdir(parents=True, exist_ok=True)")
C1 = sub(C1, "# 4. Labels (위치 3 × 고장 3) — Imbalance/Modulation 스피커는 0",
         "# 4. Labels — 3x3: (위치 3 × 고장 3), 3x1: (위치 3 × 고장 여부 1)")
C1 = sub(C1, """    return np.array([[[float(r['classes'][p] == c) for c in TY] for p in range(3)] for r in records], np.float32)
""", """    return np.array([[[float(r['classes'][p] == c) for c in TY] for p in range(3)] for r in records], np.float32)

def labels_for(mode, Y3): return Y3 if mode == '3x3' else Y3.max(2, keepdims=True)
""")
C1 = sub(C1, '''    """지도 3장 → 가지 3개 → 각도 방향 1D CNN(원형) → FC → 9 logits (위치 3 × 고장 3)."""
    def __init__(self, dropout=0.3):
        super().__init__(); self.branches''', '''    """지도 3장 → 가지 3개 → 각도 방향 1D CNN(원형) → FC → 위치 3 × n_types logits (3x3: 9개, 3x1: 3개)."""
    def __init__(self, dropout=0.3, n_types=3):
        super().__init__(); self.nt = n_types; self.branches''')
C1 = sub(C1, "self.out = nn.Linear(64 * CONFIG['angle_bins'], 9)", "self.out = nn.Linear(64 * CONFIG['angle_bins'], 3 * n_types)")
C1 = sub(C1, "return self.out(self.drop(h.flatten(1))).view(-1, 3, 3)", "return self.out(self.drop(h.flatten(1))).view(-1, 3, self.nt)")
C1 = sub(C1, """def report(P, Y, fc, date, thr):""", '''def sector_auc(P, Y):
    """섹터(스피커) 단위: 이 스피커가 고장인가 — 검증(조기 종료)용 (3x1)."""
    return float(roc_auc_score(Y.max(2).ravel(), P.max(2).ravel()))

def sector_report(Ps, Ys, fc, date, thr):
    """Ps: (N, 3) 섹터 고장 확률, Ys: (N, 3) 0/1. 고장 개수별 정상·고장 섹터 판별, 순위, 정상 녹음 오탐."""
    pred = Ps >= thr; S = dict(sector_auc_all=float(roc_auc_score(Ys.ravel(), Ps.ravel())))
    m = (fc == 1) | (fc == 2); S['sector_auc_1_2'] = float(roc_auc_score(Ys[m].ravel(), Ps[m].ravel()))
    for q in range(4):
        k = fc == q; H = Ys[k] == 0; Fm = Ys[k] > 0; p = pred[k]
        S[f'fc{q}'] = dict(n_rec=int(k.sum()), healthy_sectors=int(H.sum()), healthy_correct=int((~p[H]).sum()),
                           fault_sectors=int(Fm.sum()), fault_correct=int(p[Fm].sum()), sector_acc=float((p == Fm).mean()),
                           all3_correct=float((p == Fm).all(1).mean()))
    k1 = np.where(fc == 1)[0]; S['fc1_fault_sector_ranked_first'] = float(np.mean([Ys[i, Ps[i].argmax()] for i in k1]))
    S['healthy_rec_flagged'] = {d: float(pred[(fc == 0) & (date == d)].any(1).mean()) for d in sorted(set(date[fc == 0]))}
    return S

def report(P, Y, fc, date, thr):''')
C1 = sub(C1, "def train_fold(X, Y, tr, va, seed, run_dir):", "def train_fold(X, Y, tr, va, seed, run_dir):\n    \"\"\"Y: (N, 3, n_types). n_types=3 → 3x3, 1 → 3x1. 검증 지표: 3x3은 녹음 AUC(고장 종류별), 3x1은 섹터 AUC.\"\"\"")
C1 = sub(C1, "dtype=torch.float32, device=device).view(1, 3, 3)", "dtype=torch.float32, device=device).view(1, 3, -1)")
C1 = sub(C1, "model = FaultBandNet(CONFIG['dropout']).to(device)", "model = FaultBandNet(CONFIG['dropout'], Y.shape[2]).to(device)")
C1 = sub(C1, "val = recording_auc(predict(va), Y[va])", "val = (recording_auc if Y.shape[2] == 3 else sector_auc)(predict(va), Y[va])")
C1 = C1[:C1.index("# ==========================================\n# 8. Multi-Fold Loop Start!")] + r'''# ==========================================
# 8. 원본 음원 비중복 반복 분할
# ==========================================
def make_repeats(records):
    """원본 파일(세 스피커 모두) 기준 분할 + 고장 개수(0/1/2/3곳)별 같은 비율.
    완전한 k-fold는 불가능(정상 원본이 최대 12번, 여러 위치에 재사용). 매 회차:
    (1) 시험: 고장 개수마다 할당량 = round(test_rate × 그 개수의 녹음 수). 덜 시험된 녹음부터 그 원본 파일을 시험 집합 T에
        넣되, 세 원본이 모두 T 안인 녹음(=시험) 수가 할당량을 넘지 않을 때만 넣는다.
    (2) 학습 풀: T의 파일이 하나도 없는 녹음. 고장 개수마다 남은 비율이 다르므로 가장 작은 비율에 맞춰 무작위로 덜어낸다.
    (3) 검증: 학습 풀에서 고장 개수마다 validation_fraction씩 뗀다."""
    files = sorted({x for r in records for x in r['source_files']}); fi = {f: i for i, f in enumerate(files)}
    E = np.array([[fi[x] for x in r['source_files']] for r in records]); n = len(records)
    fc = FC; nf = np.bincount(fc, minlength=4); quota = np.round(CONFIG['test_rate'] * nf).astype(int)
    rng = np.random.default_rng(CONFIG['split_seed']); tested = np.zeros(n, int); reps = []
    for rep in range(CONFIG['repeats']):
        inT = np.zeros(len(files), bool); w = 1.0 / (1 + tested) ** 2; cnt = np.zeros(4, int)
        for e in rng.choice(n, n, replace=False, p=w / w.sum()):
            if (cnt >= quota).all(): break
            if cnt[fc[e]] >= quota[fc[e]] or inT[E[e]].all(): continue
            t2 = inT.copy(); t2[E[e]] = True; c2 = np.bincount(fc[t2[E].all(1)], minlength=4)
            if (c2 <= quota).all(): inT, cnt = t2, c2
        te = np.where(inT[E].all(1))[0]; pool = np.where(~inT[E].any(1))[0]
        assert not (set(E[te].ravel()) & set(E[pool].ravel()))                  # 원본 파일 겹침 0
        frac = (np.bincount(fc[pool], minlength=4) / nf).min(); keep = np.floor(frac * nf).astype(int)
        tr, va = [], []
        for q in range(4):
            sel = rng.permutation(rng.choice(pool[fc[pool] == q], keep[q], replace=False)); nv = int(round(CONFIG['validation_fraction'] * keep[q]))
            va.append(sel[:nv]); tr.append(sel[nv:])
        tested[te] += 1; reps.append(dict(repeat=rep, train=np.sort(np.concatenate(tr)), validation=np.sort(np.concatenate(va)), test=te))
    return reps

# ==========================================
# 9. Multi-Repeat Loop Start!
# ==========================================
print('Reading dataset (read-only):', CONFIG['data_root'], flush=True)
ALL = discover(CONFIG['data_root']); X_ALL = build_features(ALL)
keep = np.array([not (set(r['classes']) & set(CONFIG['exclude_classes'])) for r in ALL])
RECORDS = [r for r, k in zip(ALL, keep) if k]; X = [x[keep] for x in X_ALL]; N = len(RECORDS)
Y3 = make_labels(RECORDS); DATE = np.array([r['date'] for r in RECORDS])
FC = Y3.max(2).sum(1).astype(int)                                            # 고장 개수 = Whine/PWM/Wind 스피커 수 (Imbalance·Modulation은 정상 취급)
KIND0 = np.array(['Imb/Mod only' if f == 0 and set(r['classes']) & set(CONFIG['healthy_like']) else f'healthy {d}' for f, r, d in zip(FC, RECORDS, DATE)])
print(f'Recordings: {N} | Whine/PWM/Wind count 0/1/2/3: {np.bincount(FC).tolist()} | count-0 kinds {dict(zip(*np.unique(KIND0[FC == 0], return_counts=True)))} '
      f'| fault speakers {dict(zip(TY, Y3.sum((0, 1)).astype(int).tolist()))}')

REPEATS = make_repeats(RECORDS)
COUNT = np.zeros(N); [np.add.at(COUNT, r['test'], 1) for r in REPEATS]; TESTED = np.where(COUNT > 0)[0]
print(f'{len(REPEATS)} repeats | train {min(len(r["train"]) for r in REPEATS)}–{max(len(r["train"]) for r in REPEATS)} '
      f'| test {min(len(r["test"]) for r in REPEATS)}–{max(len(r["test"]) for r in REPEATS)} | tested at least once: {len(TESTED)}/{N}')
for q in range(4): print(f'  {q} fault: train / val / test per repeat (mean) = ' + ' / '.join(f"{np.mean([(FC[r[k]] == q).sum() for r in REPEATS]):.1f}" for k in ('train', 'validation', 'test')) + f'  of {(FC == q).sum()}')
write_json(OUT / 'repeats.json', [{'repeat': r['repeat'], **{k: [RECORDS[i]['uid'] for i in r[k]] for k in ('train', 'validation', 'test')}} for r in REPEATS])

MODE = CONFIG['output_mode']; Y = labels_for(MODE, Y3)
if CONFIG['train']:
    if CONFIG['device'] == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable.')
    S = np.zeros(Y.shape); REP_PROBS = []
    for r in REPEATS:
        tr, va, te = r['train'], r['validation'], r['test']; run_dir = OUT / MODE / f"repeat{r['repeat']:02d}"; run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{MODE} repeat {r['repeat']}] train {len(tr)} / val {len(va)} / test {len(te)}", flush=True)
        predict = train_fold(X, Y, tr, va, CONFIG['seed_base'] + r['repeat'], run_dir)
        p = predict(te); S[te] += p; REP_PROBS.append(p)
    PROBS = (S / np.maximum(COUNT, 1)[:, None, None]).astype(np.float32)
    np.save(OUT / f'test_probabilities_{MODE}.npy', PROBS)
    np.save(OUT / f'repeat_probabilities_{MODE}.npy', np.array(REP_PROBS, dtype=object), allow_pickle=True)
else:
    PROBS = np.load(OUT / f'test_probabilities_{MODE}.npy')
    REP_PROBS = list(np.load(OUT / f'repeat_probabilities_{MODE}.npy', allow_pickle=True))
print('Training done. Raw dataset was read only.')
'''

C2 = r'''# ==========================================
# 1. Test Evaluation (시험된 녹음만, 회차 평균 확률)
# ==========================================
THR = CONFIG['threshold']; T = TESTED; YS = Y3.max(2)                       # YS: (N, 3) 섹터 고장 여부
PS = PROBS.max(2)                                                            # 섹터 고장 확률 (3x3: 세 고장 중 최대)
rows = []
for i in T:
    r = RECORDS[i]
    for p, pos in enumerate(POS):
        row = dict(recording_id=r['uid'], date=r['date'], fault_count=int(FC[i]), times_tested=int(COUNT[i]), position=pos, gt_class=r['classes'][p])
        if MODE == '3x3':
            row.update({f'prob_{c}': round(float(PROBS[i, p, j]), 4) for j, c in enumerate(TY)})
            row['predicted'] = next((TY[j] for j in np.argsort(-PROBS[i, p]) if PROBS[i, p, j] >= THR), 'Healthy')
        else:
            row['prob_fault'] = round(float(PROBS[i, p, 0]), 4); row['predicted'] = 'Fault' if PROBS[i, p, 0] >= THR else 'Healthy'
        rows.append(row)
write_csv(OUT / f'test_predictions_{MODE}.csv', rows)

# ==========================================
# 2. 회차별 지표 (회차마다 시험 녹음 35개라 흔들림이 큼)
# ==========================================
v = np.array([roc_auc_score(YS[r['test']].ravel(), p.max(2).ravel()) for r, p in zip(REPEATS, REP_PROBS) if 0 < YS[r['test']].mean() < 1])
print(f'{MODE} sector AUC per repeat: mean {v.mean():.3f} ± {v.std():.3f} (min {v.min():.3f}, max {v.max():.3f}, {len(v)} repeats)')

# ==========================================
# 3. Grand Summary
# ==========================================
SECTOR = sector_report(PS[T], YS[T], FC[T], DATE[T], THR)
DATE0 = np.where(KIND0 == 'Imb/Mod only', 'imb-mod', DATE)                  # '정상 11/09·11/02' 오탐에서 Imbalance·Modulation만 있는 녹음은 뺌
TYPES = report(PROBS[T], Y3[T], FC[T], DATE0[T], THR) if MODE == '3x3' else None
write_json(OUT / f'summary_{MODE}.json', dict(config=CONFIG, n_tested=int(len(T)), times_tested_mean=float(COUNT[T].mean()), sector=SECTOR, types=TYPES))
print(f'\n=== {MODE}: 섹터 1개 정상/고장 판별 (시험된 녹음 {len(T)}개, 원본 음원 비중복) ===')
fr = lambda a, b: f'{a}/{b} = {a / b:.2f}' if b else '-'
print(f"{'fault count':<12}{'rec.':>6}{'healthy sectors right':>24}{'fault sectors right':>22}{'sector acc':>12}{'all 3 right':>13}")
for q in range(4):
    s = SECTOR[f'fc{q}']
    print(f"{q:<12}{s['n_rec']:>6}{fr(s['healthy_correct'], s['healthy_sectors']):>24}{fr(s['fault_correct'], s['fault_sectors']):>22}{s['sector_acc']:>12.2f}{s['all3_correct']:>13.2f}")
print(f"sector AUC: all {SECTOR['sector_auc_all']:.3f} | within 1-/2-fault rec. {SECTOR['sector_auc_1_2']:.3f}")
print(f"1-fault rec.: fault sector ranked 1st {SECTOR['fc1_fault_sector_ranked_first']:.3f} (random 0.33)")
flag = PS[T] >= THR; k0 = FC[T] == 0
print('count-0 recordings flagged (any sector):', {k: f"{int(flag[k0 & (KIND0[T] == k)].any(1).sum())}/{int((k0 & (KIND0[T] == k)).sum())}" for k in sorted(set(KIND0[T][k0]))})
if MODE == '3x3':
    f2 = lambda x: '  -  ' if x is None else f'{x:.2f}'
    print('\n=== 3x3: 고장 종류별 (녹음 단위) ===')
    print(f"{'fault':<7}{'AUC':>6}{'in-block':>10}{'detected':>10}{'n':>6} | false alarms: {'healthy 11/09':>14}{'healthy 11/02':>15}{'other fault rec.':>18} | position")
    for c in TY:
        s = TYPES[c]; print(f"{c:<7}{s['auc']:>6.3f}{f2(s['in_block_auc']):>10}{s['detected']:>10.2f}{s['n']:>6} | {f2(s['fa_healthy_1109']):>28}{f2(s['fa_healthy_1102']):>15}{s['fa_fault_rec_without']:>18.2f} | {f2(s['position_correct'])}")
    print('1-fault recordings, exactly the right type:', round(TYPES['type_exact_1fault'], 3))
'''

C3 = r'''# 결과 그림 1 (Nature style): 고장 개수별 섹터 판별 + (3x3) 고장 종류별 검출·오탐 / (3x1) 섹터 지표
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 9,
                     'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8})
COL = {'Whine': '#4C72B0', 'PWM': '#DD8452', 'Wind': '#55A868'}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={'width_ratios': [1.4, 1]})
xs = [(q, k) for q in range(4) for k in ('healthy', 'fault') if SECTOR[f'fc{q}'][f'{k}_sectors']]
v = [SECTOR[f'fc{q}'][f'{k}_correct'] / SECTOR[f'fc{q}'][f'{k}_sectors'] for q, k in xs]
ax1.bar(range(len(xs)), v, 0.6, color=['#9ad9c5' if k == 'healthy' else '#F4A3A3' for q, k in xs], edgecolor='k', linewidth=0.6)
for a, b in enumerate(v): ax1.text(a, b + .02, f'{b:.2f}', ha='center', fontsize=7)
ax1.set_xticks(range(len(xs))); ax1.set_xticklabels([f'{q} fault\n{k}' for q, k in xs], fontsize=7); ax1.set_ylim(0, 1.1)
ax1.set_ylabel('Sectors judged correctly'); ax1.set_title(f'{MODE}: sector level, by fault count of the recording', fontsize=9)
if MODE == '3x3':
    x = np.arange(3); w2 = 0.26
    ax2.bar(x - w2, [TYPES[c]['detected'] for c in TY], w2, color=[COL[c] for c in TY], edgecolor='k', linewidth=0.6, label='Detected')
    ax2.bar(x, [TYPES[c]['fa_fault_rec_without'] for c in TY], w2, color=[COL[c] for c in TY], alpha=0.45, edgecolor='k', linewidth=0.6, hatch='//', label='False alarm (other-fault rec.)')
    ax2.bar(x + w2, [TYPES[c]['fa_healthy_1102'] or 0 for c in TY], w2, color='white', edgecolor='k', linewidth=0.6, label='False alarm (healthy 11/02)')
    ax2.set_xticks(x); ax2.set_xticklabels(TY); ax2.legend(frameon=False, fontsize=6, loc='upper right'); ax2.set_title('Recording level, per fault type', fontsize=9)
else:
    vals = [SECTOR['sector_auc_1_2'], SECTOR['fc1_fault_sector_ranked_first'], SECTOR['fc1']['all3_correct'], SECTOR['fc2']['all3_correct']]
    ax2.bar(range(4), vals, 0.6, color='#8172B3', edgecolor='k', linewidth=0.6); ax2.axhline(1 / 3, color='gray', ls='--', lw=0.8)
    for a, b in enumerate(vals): ax2.text(a, b + .02, f'{b:.2f}', ha='center', fontsize=7)
    ax2.set_xticks(range(4)); ax2.set_xticklabels(['Sector AUC\n(1-/2-fault)', 'Fault sector\nranked 1st', 'All 3 right\n(1 fault)', 'All 3 right\n(2 faults)'], fontsize=7)
    ax2.set_title('Position (sector) level', fontsize=9)
ax2.set_ylim(0, 1.1)
plt.tight_layout(); (OUT / 'figures').mkdir(exist_ok=True)
plt.savefig(OUT / f'figures/Fig_Sector_Summary_{MODE}.pdf'); plt.savefig(OUT / f'figures/Fig_Sector_Summary_{MODE}.png', dpi=200); plt.show()
'''

C4 = r'''# 결과 그림 2: 시험 녹음 예시 — 입력 지도 3장(각도 × 주파수)과 출력 확률(위치 × 고장)
import matplotlib.pyplot as plt
picks = [T[np.where(FC[T] == q)[0][0]] for q in range(4) if (FC[T] == q).any()]
cols = TY if MODE == '3x3' else ['fault']
fig, axes = plt.subplots(len(picks), 4, figsize=(9.5, 2.2 * len(picks)), gridspec_kw={'width_ratios': [1, 1, 1, 0.9 if MODE == '3x3' else 0.5]})
ext = [(CONFIG['wind_band'][0] / 1000, CONFIG['wind_band'][1] / 1000), (CONFIG['pwm_band'][0] / 1000, CONFIG['pwm_band'][1] / 1000), (-150, 150)]
names = ['Wind map (kHz)', 'PWM map (kHz)', 'Whine map (Hz from order 16)']
for r, i in enumerate(picks):
    for j in range(3):
        ax = axes[r, j]; ax.imshow(X[j][i], origin='lower', aspect='auto', cmap='magma', extent=[ext[j][0], ext[j][1], 0, 360], vmin=-3, vmax=12)
        ax.set_yticks([0, 90, 180, 270, 360]); ax.tick_params(labelsize=6)
        if r == 0: ax.set_title(names[j], fontsize=8)
        if j == 0: ax.set_ylabel('Mic angle (°)', fontsize=7)
    M = PROBS[i]; ax = axes[r, 3]; ax.imshow(M, cmap='Reds', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=6)
    ax.set_yticks(range(3)); ax.set_yticklabels([f'{p} ({c})' for p, c in zip(POS, RECORDS[i]['classes'])], fontsize=6)
    for a in range(3):
        for b in range(len(cols)): ax.text(b, a, f'{M[a, b]:.2f}', ha='center', va='center', fontsize=6, color='white' if M[a, b] > 0.6 else 'black')
    if r == 0: ax.set_title(f'{MODE} output (true class)', fontsize=8)
plt.tight_layout(); plt.savefig(OUT / f'figures/Fig_Test_Examples_{MODE}.pdf'); plt.savefig(OUT / f'figures/Fig_Test_Examples_{MODE}.png', dpi=200); plt.show()
'''

b_main_line = '''  (위치별 고장 여부만 내는 3×1 변형은 서브 노트북 `sub/3DAR_EV_FaultBandNet_3x1.ipynb`)
'''
MD3x1 = MD.replace(b_main_line, '', 1).replace('**3×3 = 위치 3(앞/뒤/오른쪽) × 고장 3(Whine/PWM/Wind) 확률**', '**3×1 = 위치 3(앞/뒤/오른쪽) × 고장 여부 확률**').replace('# 3DAR_EV FaultBandNet — 고장 대역 × 마이크 각도 지도로 고장 위치(·종류) 판정',
                   '# [서브] 3DAR_EV FaultBandNet 3×1 — 위치별 고장 여부만 판정\n\n'
                   '주 노트북(`../3DAR_EV_FaultBandNet.ipynb`, 3×3)과 입력·신경망·분할·학습이 같고 **마지막 층만 2304 → 3**(위치별 고장 여부)이다. '
                   '라벨 = 그 스피커가 Whine/PWM/Wind면 1. 검증 지표는 섹터 AUC. 비교용 서브 실험.')
MD3x3 = MD


def build(mode, pre, md):
    c1 = C1.replace('@MODE@', mode).replace('@PRE@', pre)
    nb = dict(cells=[OLD.cell('markdown', md), OLD.cell('code', c1), OLD.cell('code', C2), OLD.cell('code', C3), OLD.cell('code', C4)],
              metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
    return json.dumps(nb, ensure_ascii=False, indent=1)


ROOT = Path(__file__).resolve().parents[2]
(ROOT / '3DAR_EV_FaultBandNet.ipynb').write_text(build('3x3', '', MD3x3), encoding='utf-8')
(ROOT / 'sub').mkdir(exist_ok=True); (ROOT / 'sub/3DAR_EV_FaultBandNet_3x1.ipynb').write_text(build('3x1', '../', MD3x1), encoding='utf-8')
print('wrote', ROOT / '3DAR_EV_FaultBandNet.ipynb', ROOT / 'sub/3DAR_EV_FaultBandNet_3x1.ipynb')
