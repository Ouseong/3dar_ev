"""Write 3DAR_EV_FaultBandNet.ipynb (penguin-notebook layout: one big train cell, one evaluation cell, figure cells).
Usage: python make_notebook.py  -> EV_3DAR_AI/3DAR_EV_FaultBandNet.ipynb
"""
from pathlib import Path
import json

MD = """# 3DAR_EV FaultBandNet — fault type and position from fault-band × microphone-angle maps

전기차 고장음을 3DAR(회전 메타구조 마이크)로 녹음한 뒤, **고장이 나타나는 대역만** 각도별로 뽑아 신경망에 넣는다.

- 입력: 녹음 1개(한 바퀴 ≈ 4 s) → 지도 3장 (마이크 각도 36칸 × 주파수)
  - Wind 3.3–6.2 kHz · PWM 5 kHz 주변 · Whine order 16 주변 (모두 녹음 안 상대값)
- 모델: 지도별 2D CNN → 각도 방향 1D CNN(원형) → FC → 출력 9개 = 위치 3(앞/뒤/오른쪽) × 고장 3(Whine/PWM/Wind)
- 평가: 오른쪽 스피커 원본 파일 단위 5-fold (학습에 쓰지 않은 녹음만 판정)
- Imbalance · Modulation은 녹음에서 성분이 사라져 대상에서 제외(라벨 0)

셀 구성(펭귄 노트북과 같은 순서): **셀 1** 설정·데이터·특징·모델·학습 → **셀 2** 평가·요약 → **셀 3–4** 그림.
데이터(측정 WAV, KU Leuven 원본)는 저장소에 포함하지 않는다. `CONFIG['data_root']`만 바꾸면 된다.
"""

CELL1 = r'''# ==========================================
# 1. Configuration
# ==========================================
import csv, json, time, random, hashlib, os
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import soundfile as sf
from scipy import signal
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import torch
from torch import nn
import torch.nn.functional as F

CONFIG = {
    # --- 경로 ---
    'data_root': '/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV',   # 측정_목록.csv 가 있는 폴더 (읽기 전용)
    'out_dir': 'runs/nb_faultbandnet_20261124',                            # 이 노트북의 모든 결과
    'device': 'cuda', 'train': True, 'workers': 10,
    # --- 녹음 / 위치 ---
    'sample_rate': 48000, 'positions': ['FRONT', 'REAR', 'RIGHT'], 'fault_types': ['Whine', 'PWM', 'Wind'],
    # --- 특징: 고장 대역 × 마이크 각도 지도 ---
    'n_fft': 8192, 'hop_length': 480, 'angle_bins': 36,
    'wind_band': [3300, 6200], 'wind_bin_hz': 50, 'wind_flanks': [4000, 5500],     # 양옆(<4.0, >5.5 kHz)에 맞춘 직선을 뺌
    'pwm_band': [4800, 5200], 'pwm_bins': 68, 'pwm_line_exclude': [4975, 5025],    # Wind 지도에서는 5 kHz 선을 제외
    'whine_order': 16, 'whine_halfwidth_bins': 25,                                  # order 16 주파수 ± 25칸(≈150 Hz)
    'rpm_refine_orders': [4, 8, 12, 16],                                            # 정상 order가 가장 선명해지도록 추정 RPM 보정
    # --- 학습 ---
    'folds': 5, 'validation_fraction': 0.15, 'seed_base': 100,
    'batch_size': 32, 'epochs': 80, 'patience': 15, 'learning_rate': 1e-3, 'weight_decay': 1e-3,
    'dropout': 0.3, 'max_pos_weight': 20.0, 'threshold': 0.5, 'gpu_duty': 0.5,
}
POS, TY = CONFIG['positions'], CONFIG['fault_types']
OUT = Path(CONFIG['out_dir']); OUT.mkdir(parents=True, exist_ok=True)

# ==========================================
# 0. 보조 함수
# ==========================================
def read_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def write_csv(path, rows):
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def write_json(path, obj): Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class GpuThrottle:
    """발열 제한: 매 step 뒤 쉬어서 GPU 사용을 약 duty 비율로 맞춘다 (결과는 같다)."""
    def __init__(self, duty): self.duty = duty; self.t = time.perf_counter()
    def reset(self): self.t = time.perf_counter()
    def step(self):
        if self.duty >= 1: return
        if torch.cuda.is_available(): torch.cuda.synchronize()
        busy = time.perf_counter() - self.t; time.sleep(busy * (1 - self.duty) / max(self.duty, .05)); self.t = time.perf_counter()

# ==========================================
# 2. Parsing Dataset (측정_목록.csv + 녹음별 info CSV, 3개 스피커 녹음 1,200개)
# ==========================================
def discover(root):
    root = Path(root); recs = []
    for row in read_csv(root / '측정_목록.csv'):
        if int(row['스피커수']) != 3: continue
        p = {k: root / str(row[c]).replace('\\', '/') for k, c in [('wav', 'WAV'), ('info', '실험정보CSV'), ('angles', '각도CSV')]}
        info = {s['Speaker_Loc']: s for s in read_csv(p['info'])}
        if set(info) != set(POS) or any(s['status'] != 'OK' for s in info.values()): raise ValueError('Bad capture: ' + row['측정ID'])
        cls = [info[q]['Sound_Class'] for q in POS]
        recs.append(dict(uid=row['측정ID'], wav=str(p['wav']), angles=str(p['angles']), classes=cls,
                         fault_count=sum(c != 'Healthy' for c in cls), date=info[POS[0]]['event_time'][5:10],
                         source_files=[info[q]['Source_File'].replace('\\', '/') for q in POS],
                         rpm_intervals=[list(map(float, info[q]['Estimated_RPM_Interval'].split('~'))) for q in POS]))
    recs.sort(key=lambda r: r['uid']); return recs

# ==========================================
# 3. Feature Extraction (각도 정렬, RPM 보정, 고장 대역 지도 3장)
# ==========================================
def angle_track(path, duration):
    rows = read_csv(path); t = np.array([float(r['elapsed_sec']) for r in rows]); a = np.array([float(r['angle_deg']) for r in rows])
    keep = np.isfinite(t) & np.isfinite(a) & (t >= 0) & (t <= duration); t, a = t[keep], a[keep]
    t, u = np.unique(t, return_index=True); a = a[u]
    inc = np.r_[True, a[1:] > np.maximum.accumulate(a[:-1])]; t, a = t[inc], a[inc]
    k = (t > 0) & (t < duration) & (a > 0) & (a < 360)
    return np.r_[0.0, t[k], duration], np.r_[0.0, a[k], 360.0]

def make_rpm(r0, r1, c, dur):
    return lambda t: (lambda u: r0 + (r1 - r0) * u + c * (r1 - r0) * u * (1 - u))(np.clip(t / dur, 0, 1))

def order_spectrum(x, fs, rpm_fn, spr=512):
    """회전각 기준 재샘플링(order tracking) → (order, power dB)."""
    b, a = signal.butter(6, 10000, fs=fs); x = signal.filtfilt(b, a, x.astype(np.float64)); t = np.arange(len(x)) / fs
    rev = np.concatenate([[0], np.cumsum(rpm_fn(t[:-1]) / 60.0 / fs)]); m = int(rev[-1] * spr)
    xr = np.interp(np.arange(m) / spr, rev, x); P = np.abs(np.fft.rfft(xr * np.hanning(m))) ** 2
    return np.fft.rfftfreq(m, 1.0 / spr), 10 * np.log10(P + 1e-30)

def peak_over_bg(o, P, c, half=0.02, bg=(0.15, 0.6)):
    d = np.abs(o - c); return float(P[d <= half].max() - np.median(P[(d >= bg[0]) & (d <= bg[1])]))

def refine_rpm(audio, fs, r0, r1, dur):
    """추정 RPM(재생 구간 기반)의 시작/끝/곡률을, 정상 order 선이 가장 선명해지도록 좌표 탐색으로 보정 (라벨 미사용)."""
    def sharp(q):
        o, P = order_spectrum(audio, fs, make_rpm(r0 * (1 + q[0]), r1 * (1 + q[1]), q[2], dur))
        return float(np.mean([peak_over_bg(o, P, k) for k in CONFIG['rpm_refine_orders']]))
    p = np.zeros(3); best = sharp(p)
    for step in (0.01, 0.0025):
        improved = True
        while improved:
            improved = False
            for j in range(3):
                for d in (-step, step):
                    q = p.copy(); q[j] += d * (10 if j == 2 else 1); v = sharp(q)
                    if v > best + 1e-3: best, p, improved = v, q, True
    return p

def extract_maps(rec):
    """녹음 1개 → wind (36×58), pwm (36×68), whine (36×51). 모두 녹음 안 상대값(dB)."""
    c = CONFIG; nf, na = c['n_fft'], c['angle_bins']
    a, fs = sf.read(rec['wav'], dtype='float32'); dur = len(a) / fs; at, aa = angle_track(rec['angles'], dur)
    fr = np.fft.rfftfreq(nf, 1 / fs)
    _, t, p = signal.spectrogram(a.astype(np.float64), fs=fs, window='hann', nperseg=nf, noverlap=nf - c['hop_length'], scaling='spectrum', mode='psd')
    db = 10 * np.log10(p + 1e-30); ab = np.clip((np.interp(t, at, aa) // (360 / na)).astype(int), 0, na - 1)
    # RPM (추정 구간의 세 스피커 중앙값 → 보정)
    r0, r1 = np.median(np.array(rec['rpm_intervals']), axis=0); q = refine_rpm(a, fs, r0, r1, dur)
    rpm = make_rpm(r0 * (1 + q[0]), r1 * (1 + q[1]), q[2], dur)(t)
    # Wind: 50 Hz 칸, 5 kHz 선 제외, 양옆 직선 제거
    edg = np.arange(c['wind_band'][0], c['wind_band'][1] + 1, c['wind_bin_hz']); cen = (edg[:-1] + edg[1:]) / 2
    ok = ~((fr > c['pwm_line_exclude'][0]) & (fr < c['pwm_line_exclude'][1]))
    wind = np.stack([10 * np.log10(p[(fr >= lo) & (fr < hi) & ok].mean(0) + 1e-30) for lo, hi in zip(edg[:-1], edg[1:])])
    flk = (cen < c['wind_flanks'][0]) | (cen > c['wind_flanks'][1]); pm = np.vander(cen / 1000, 2) @ np.linalg.pinv(np.vander(cen[flk] / 1000, 2))
    # PWM: 5 kHz 주변 고해상도
    psel = np.where((fr >= c['pwm_band'][0]) & (fr < c['pwm_band'][1]))[0][:c['pwm_bins']]
    # Whine: order 16 주파수를 따라가는 좁은 띠
    off = np.arange(-c['whine_halfwidth_bins'], c['whine_halfwidth_bins'] + 1)
    wl = db[np.rint(c['whine_order'] * rpm / 60 / fr[1]).astype(int)[None, :] + off[:, None], np.arange(len(t))[None, :]]
    W = np.zeros((na, len(cen)), np.float32); Pm = np.zeros((na, len(psel)), np.float32); H = np.zeros((na, len(off)), np.float32)
    for i in range(na):
        m = ab == i
        if not m.any(): continue
        w = wind[:, m].mean(1); W[i] = w - pm @ w[flk]                    # 프레임별 dB 평균(순간 잡음에 강함)
        x = db[psel][:, m].mean(1); Pm[i] = x - np.median(x)
        h = wl[:, m].mean(1); H[i] = h - np.median(h)
    return W, Pm, H

def build_features(records):
    key = hashlib.sha256(json.dumps({k: CONFIG[k] for k in ['n_fft', 'hop_length', 'angle_bins', 'wind_band', 'wind_bin_hz', 'wind_flanks',
          'pwm_band', 'pwm_bins', 'pwm_line_exclude', 'whine_order', 'whine_halfwidth_bins', 'rpm_refine_orders']}, sort_keys=True).encode()).hexdigest()[:16]
    cache = OUT / f'features_{key}.npz'
    if cache.exists():
        z = np.load(cache)
        if list(z['uid']) == [r['uid'] for r in records]: print('Feature cache reused:', cache.name); return [z['wind'], z['pwm'], z['whine']]
    print(f'Extracting features for {len(records)} recordings ...', flush=True)
    with ProcessPoolExecutor(CONFIG['workers']) as ex: res = list(ex.map(extract_maps, records, chunksize=8))
    maps = [np.stack([r[i] for r in res]) for i in range(3)]
    np.savez(cache, uid=np.array([r['uid'] for r in records]), wind=maps[0], pwm=maps[1], whine=maps[2]); return maps

# ==========================================
# 4. Labels (위치 3 × 고장 3) — Imbalance/Modulation 스피커는 0
# ==========================================
def make_labels(records):
    return np.array([[[float(r['classes'][p] == c) for c in TY] for p in range(3)] for r in records], np.float32)

# ==========================================
# 5. Model Architecture
# ==========================================
class Branch(nn.Module):
    """지도 1장(각도 × 주파수)용 2D CNN. 각도 축은 원형 패딩, 주파수 축만 풀링 → (B, 64, 각도 36)."""
    def __init__(self):
        super().__init__()
        self.conv = nn.ModuleList([nn.Conv2d(a, b, 3) for a, b in ((1, 16), (16, 32), (32, 32))])
        self.norm = nn.ModuleList([nn.GroupNorm(4, b) for b in (16, 32, 32)])
    def forward(self, x):
        for i, (c, g) in enumerate(zip(self.conv, self.norm)):
            x = F.gelu(g(c(F.pad(F.pad(x, (1, 1, 0, 0)), (0, 0, 1, 1), mode='circular'))))
            if i < 2: x = F.max_pool2d(x, (1, 2))
        return torch.cat([x.mean(3), x.amax(3)], 1)

class FaultBandNet(nn.Module):
    """지도 3장 → 가지 3개 → 각도 방향 1D CNN(원형) → FC → 9 logits (위치 3 × 고장 3)."""
    def __init__(self, dropout=0.3):
        super().__init__(); self.branches = nn.ModuleList([Branch() for _ in range(3)])
        self.a1 = nn.Conv1d(192, 64, 5, padding=2, padding_mode='circular'); self.a2 = nn.Conv1d(64, 64, 5, padding=2, padding_mode='circular')
        self.drop = nn.Dropout(dropout); self.out = nn.Linear(64 * CONFIG['angle_bins'], 9)
    def forward(self, xs):
        h = torch.cat([b(x) for b, x in zip(self.branches, xs)], 1)
        h = F.gelu(self.a2(self.drop(F.gelu(self.a1(h)))))
        return self.out(self.drop(h.flatten(1))).view(-1, 3, 3)

# ==========================================
# 6. Evaluation Functions
# ==========================================
def recording_auc(P, Y):
    v = []
    for c in range(3):
        y = Y[:, :, c].max(1)
        if 0 < y.mean() < 1: v.append(roc_auc_score(y, P[:, :, c].max(1)))
    return float(np.mean(v))

def report(P, Y, fc, date, thr):
    """P, Y: (N, 위치 3, 고장 3). 고장별 검출/오탐, 종류, 위치(섹터) 지표."""
    S = {}
    for c, name in enumerate(TY):
        s = P[:, :, c].max(1); has = Y[:, :, c].max(1) > 0; pred = s >= thr; num = den = 0
        for b in sorted({(d, f) for d, f in zip(date, fc) if f in (1, 2)}):      # 같은 날짜·고장 개수 블록 안 비교
            k = (date == b[0]) & (fc == b[1])
            if has[k].any() and (~has[k]).any(): w = has[k].sum() * (~has[k]).sum(); num += roc_auc_score(has[k], s[k]) * w; den += w
        one = Y[:, :, c].sum(1) == 1; h1109 = (fc == 0) & (date == '11-09'); h1102 = (fc == 0) & (date == '11-02')
        S[name] = dict(auc=float(roc_auc_score(has, s)) if 0 < has.mean() < 1 else None, in_block_auc=float(num / den) if den else None,
                       detected=float(pred[has].mean()) if has.any() else None, n=int(has.sum()),
                       fa_healthy_1109=float(pred[h1109].mean()) if h1109.any() else None, fa_healthy_1102=float(pred[h1102].mean()) if h1102.any() else None,
                       fa_fault_rec_without=float(pred[(fc > 0) & ~has].mean()),
                       position_correct=float((P[one][:, :, c].argmax(1) == Y[one][:, :, c].argmax(1)).mean()) if one.any() else None)
    pr = P.max(1) >= thr; hs = Y.max(1) > 0; anyp = pr.any(1); det = hs.any(1); k1 = (fc == 1) & det
    S['any_fault'] = dict(recall=float(anyp[det].mean()), flagged_all_healthy=float(anyp[fc == 0].mean()) if (fc == 0).any() else None,
                          flagged_only_imbalance_modulation=float(anyp[(fc > 0) & ~det].mean()) if ((fc > 0) & ~det).any() else None)
    S['type_exact_1fault'] = float((pr[k1] == hs[k1]).all(1).mean()) if k1.any() else None
    ss = P.max(2); sy = Y.max(2); m = (fc == 1) | (fc == 2)
    S['sector'] = dict(auc_within_1_2=float(roc_auc_score(sy[m].ravel(), ss[m].ravel())),
                       fc1_fault_sector_ranked_first=float(np.mean([sy[i, ss[i].argmax()] for i in np.where(k1)[0]])) if k1.any() else None,
                       all3_exact={f'fc{q}': float(((ss[fc == q] >= thr) == (sy[fc == q] > 0)).all(1).mean()) for q in range(4) if (fc == q).any()})
    return S

# ==========================================
# 7. Train Loop
# ==========================================
def train_fold(X, Y, tr, va, seed, run_dir):
    device = torch.device(CONFIG['device']); seed_all(seed)
    mu = [float(x[tr].mean()) for x in X]; sd = [float(x[tr].std()) for x in X]                 # 정규화: train 녹음 통계만
    Z = [torch.from_numpy(((x - m) / s).astype(np.float32))[:, None] for x, m, s in zip(X, mu, sd)]; Yt = torch.from_numpy(Y)
    pos = Y[tr].reshape(len(tr), -1).sum(0)
    pw = torch.tensor(np.clip((len(tr) - pos) / np.maximum(pos, 1), 1, CONFIG['max_pos_weight']), dtype=torch.float32, device=device).view(1, 3, 3)
    model = FaultBandNet(CONFIG['dropout']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    g = torch.Generator().manual_seed(seed); throttle = GpuThrottle(CONFIG['gpu_duty'])
    def predict(ix):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(ix), 128): b = ix[i:i + 128]; out.append(torch.sigmoid(model([q[b].to(device) for q in Z])).cpu().numpy())
        return np.concatenate(out)
    best = (-1, None, 0); history = []
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train(); perm = torch.from_numpy(tr)[torch.randperm(len(tr), generator=g)]; total = 0.0; throttle.reset()
        for i in range(0, len(perm), CONFIG['batch_size']):
            b = perm[i:i + CONFIG['batch_size']]
            loss = F.binary_cross_entropy_with_logits(model([q[b].to(device) for q in Z]), Yt[b].to(device), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); total += float(loss) * len(b); throttle.step()
        val = recording_auc(predict(va), Y[va]); history.append(dict(epoch=epoch, train_loss=total / len(tr), val_auc=val))
        if val > best[0]: best = (val, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch)
        if epoch - best[2] >= CONFIG['patience']: break
    model.load_state_dict(best[1]); print(f'   best epoch {best[2]}  val AUC {best[0]:.3f}', flush=True)
    torch.save(dict(state_dict=best[1], mu=mu, sd=sd, config=dict(CONFIG), epoch=best[2]), run_dir / 'faultbandnet_best.pt')
    write_json(run_dir / 'history.json', history)
    return predict

# ==========================================
# 8. Multi-Fold Loop Start!
# ==========================================
print('Reading dataset (read-only):', CONFIG['data_root'], flush=True)
RECORDS = discover(CONFIG['data_root']); N = len(RECORDS)
X = build_features(RECORDS); Y = make_labels(RECORDS)
FC = np.array([r['fault_count'] for r in RECORDS]); DATE = np.array([r['date'] for r in RECORDS])
GROUP = np.array([r['source_files'][2] for r in RECORDS])                     # 오른쪽 스피커 원본 파일
print(f'Recordings: {N} | maps: {[x.shape[1:] for x in X]} | fault speakers per type: {dict(zip(TY, Y.sum((0, 1)).astype(int).tolist()))}')

FOLDS = []
for k, (tr, te) in enumerate(GroupKFold(CONFIG['folds']).split(np.arange(N), Y[:, 2, 2], GROUP)):
    a, b = next(GroupShuffleSplit(1, test_size=CONFIG['validation_fraction'], random_state=k).split(tr, groups=GROUP[tr]))
    FOLDS.append(dict(fold=k, train=tr[a], validation=tr[b], test=te))
    assert not (set(GROUP[tr]) & set(GROUP[te]))                                # 같은 오른쪽 원본이 학습·시험에 겹치지 않음

PROBS = np.zeros((N, 3, 3), np.float32)
if CONFIG['train']:
    if CONFIG['device'] == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable.')
    for f in FOLDS:
        run_dir = OUT / f"fold{f['fold']}"; run_dir.mkdir(exist_ok=True)
        print(f"[fold {f['fold']}] train {len(f['train'])} / val {len(f['validation'])} / test {len(f['test'])}", flush=True)
        predict = train_fold(X, Y, f['train'], f['validation'], CONFIG['seed_base'] + f['fold'], run_dir)
        PROBS[f['test']] = predict(f['test'])
    np.save(OUT / 'test_probabilities.npy', PROBS)
    print('Training done. Raw dataset was read only.')
else:
    PROBS = np.load(OUT / 'test_probabilities.npy')
'''

CELL2 = r'''# ==========================================
# 1. Test Evaluation (fold별 → 전체 요약)
# ==========================================
THR = CONFIG['threshold']
rows = []
for i, r in enumerate(RECORDS):
    for p, pos in enumerate(POS):
        rows.append(dict(recording_id=r['uid'], date=r['date'], fault_count=r['fault_count'], position=pos, gt_class=r['classes'][p],
                         **{f'prob_{c}': float(PROBS[i, p, j]) for j, c in enumerate(TY)},
                         predicted=next((TY[j] for j in np.argsort(-PROBS[i, p]) if PROBS[i, p, j] >= THR), 'none')))
write_csv(OUT / 'test_predictions.csv', rows)

# ==========================================
# 2. Fold별 지표
# ==========================================
per_fold = [report(PROBS[f['test']], Y[f['test']], FC[f['test']], DATE[f['test']], THR) for f in FOLDS]
print(f"{'metric':<34}" + ''.join(f'fold{f["fold"]:>6}' for f in FOLDS) + '      mean ± std')
def line(name, get):
    v = np.array([get(s) if get(s) is not None else np.nan for s in per_fold], float)
    print(f'{name:<34}' + ''.join(f'{x:>10.3f}' for x in v) + f'   {np.nanmean(v):.3f} ± {np.nanstd(v):.3f}')
for c in TY:
    line(f'{c} AUC', lambda s: s[c]['auc']); line(f'{c} detected', lambda s: s[c]['detected'])
line('any-fault recall', lambda s: s['any_fault']['recall']); line('all-healthy flagged', lambda s: s['any_fault']['flagged_all_healthy'])
line('sector AUC (1-/2-fault rec.)', lambda s: s['sector']['auc_within_1_2']); line('fault sector ranked first (1-fault)', lambda s: s['sector']['fc1_fault_sector_ranked_first'])

# ==========================================
# 3. Grand Summary (학습에 쓰지 않은 fold 예측을 모두 합침)
# ==========================================
SUMMARY = report(PROBS, Y, FC, DATE, THR)
write_json(OUT / 'summary.json', dict(config=CONFIG, pooled=SUMMARY, per_fold=per_fold))
f2 = lambda x: '  -  ' if x is None else f'{x:.2f}'
print('\n=== Pooled out-of-fold results (1,200 recordings) ===')
print(f"{'fault':<7}{'AUC':>6}{'in-block':>10}{'detected':>10}{'n':>6} | false alarms: {'healthy 11/09':>14}{'healthy 11/02':>15}{'other fault rec.':>18} | position")
for c in TY:
    s = SUMMARY[c]; print(f"{c:<7}{s['auc']:>6.3f}{f2(s['in_block_auc']):>10}{s['detected']:>10.2f}{s['n']:>6} | {f2(s['fa_healthy_1109']):>28}{f2(s['fa_healthy_1102']):>15}{s['fa_fault_rec_without']:>18.2f} | {f2(s['position_correct'])}")
print('any fault (Whine/PWM/Wind):', {k: round(v, 3) for k, v in SUMMARY['any_fault'].items() if v is not None})
print('1-fault recordings, exactly the right type:', round(SUMMARY['type_exact_1fault'], 3))
print('sector level:', SUMMARY['sector'])
'''

CELL3 = r'''# 결과 그림 1 (Nature style): 고장별 검출률·오탐, 위치(섹터) 지표
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 9,
                     'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8})
COL = {'Whine': '#4C72B0', 'PWM': '#DD8452', 'Wind': '#55A868'}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={'width_ratios': [1.7, 1]})
x = np.arange(3); w = 0.26
ax1.bar(x - w, [SUMMARY[c]['detected'] for c in TY], w, color=[COL[c] for c in TY], edgecolor='k', linewidth=0.6, label='Detected')
ax1.bar(x, [SUMMARY[c]['fa_fault_rec_without'] for c in TY], w, color=[COL[c] for c in TY], alpha=0.45, edgecolor='k', linewidth=0.6, hatch='//', label='False alarm (other-fault rec.)')
ax1.bar(x + w, [SUMMARY[c]['fa_healthy_1109'] or 0 for c in TY], w, color='white', edgecolor='k', linewidth=0.6, label='False alarm (all-healthy rec.)')
ax1.set_xticks(x); ax1.set_xticklabels(TY); ax1.set_ylim(0, 1.05); ax1.set_ylabel('Rate'); ax1.legend(frameon=False, fontsize=7, loc='upper right'); ax1.set_title('Recording level', fontsize=9)
sec = SUMMARY['sector']; vals = [sec['auc_within_1_2'], sec['fc1_fault_sector_ranked_first'], sec['all3_exact'].get('fc1', np.nan), sec['all3_exact'].get('fc2', np.nan)]
ax2.bar(range(4), vals, 0.6, color='#8172B3', edgecolor='k', linewidth=0.6); ax2.axhline(1 / 3, color='gray', ls='--', lw=0.8)
ax2.set_xticks(range(4)); ax2.set_xticklabels(['Sector\nAUC', 'Fault sector\nranked 1st', 'All 3 exact\n(1 fault)', 'All 3 exact\n(2 faults)'], fontsize=7)
ax2.set_ylim(0, 1.05); ax2.set_title('Position (sector) level', fontsize=9)
for a, v in zip(range(4), vals): ax2.text(a, v + 0.02, f'{v:.2f}', ha='center', fontsize=7)
plt.tight_layout(); (OUT / 'figures').mkdir(exist_ok=True); plt.savefig(OUT / 'figures/Fig_Detection_Summary.pdf'); plt.savefig(OUT / 'figures/Fig_Detection_Summary.png', dpi=200); plt.show()
'''

CELL4 = r'''# 결과 그림 2: test 녹음 예시 — 입력 지도 3장(각도 × 주파수)과 출력 확률표(위치 × 고장)
# 화면 확인용: 고장 0/1/2/3곳 녹음 하나씩 (첫 fold의 test 녹음에서)
import matplotlib.pyplot as plt
te = FOLDS[0]['test']; picks = [te[np.where(FC[te] == q)[0][0]] for q in range(4) if (FC[te] == q).any()]
fig, axes = plt.subplots(len(picks), 4, figsize=(9.5, 2.2 * len(picks)), gridspec_kw={'width_ratios': [1, 1, 1, 0.9]})
ang = [0, 360]; ext = [(CONFIG['wind_band'][0] / 1000, CONFIG['wind_band'][1] / 1000), (CONFIG['pwm_band'][0] / 1000, CONFIG['pwm_band'][1] / 1000), (-150, 150)]
names = ['Wind map (kHz)', 'PWM map (kHz)', 'Whine map (Hz from order 16)']
for r, i in enumerate(picks):
    for j in range(3):
        ax = axes[r, j]; ax.imshow(X[j][i], origin='lower', aspect='auto', cmap='magma', extent=[ext[j][0], ext[j][1], 0, 360], vmin=-3, vmax=12)
        ax.set_yticks([0, 90, 180, 270, 360]); ax.tick_params(labelsize=6)
        if r == 0: ax.set_title(names[j], fontsize=8)
        if j == 0: ax.set_ylabel('Mic angle (°)', fontsize=7)
    ax = axes[r, 3]; im = ax.imshow(PROBS[i], cmap='Reds', vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(TY, fontsize=6); ax.set_yticks(range(3)); ax.set_yticklabels([f'{p} ({c})' for p, c in zip(POS, RECORDS[i]['classes'])], fontsize=6)
    for a in range(3):
        for b in range(3): ax.text(b, a, f'{PROBS[i, a, b]:.2f}', ha='center', va='center', fontsize=6, color='white' if PROBS[i, a, b] > 0.6 else 'black')
    if r == 0: ax.set_title('Output probability (true class in brackets)', fontsize=8)
plt.tight_layout(); plt.savefig(OUT / 'figures/Fig_Test_Examples.pdf'); plt.savefig(OUT / 'figures/Fig_Test_Examples.png', dpi=200); plt.show()
'''


def cell(kind, src):
    c = dict(cell_type=kind, metadata={}, source=src.splitlines(keepends=True))
    if kind == 'code': c.update(execution_count=None, outputs=[])
    return c


nb = dict(cells=[cell('markdown', MD), cell('code', CELL1), cell('code', CELL2), cell('code', CELL3), cell('code', CELL4)],
          metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'), language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
out = Path(__file__).resolve().parents[2] / '3DAR_EV_FaultBandNet.ipynb'
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8'); print('wrote', out)
