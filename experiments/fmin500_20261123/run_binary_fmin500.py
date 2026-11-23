"""FMIN 500 Hz: same as the STRICT re-run below but the log-mel input starts at 500 Hz (3DAR nominal lower limit,
RCIM 2025) instead of 0 Hz; 64 mel bands over 500-12000 Hz. New feature cache runs/fmin500_cache_20261123 (baseline cache untouched).
STRICT (source-disjoint) re-run, 2026-11-23: identical to experiments/sup_binary_20261119/run_binary.py except
out_dir and the splits, which are loaded from runs/strict_split_20261123/seed*.json (no source file shared between
train/validation/test; recordings mixing splits are dropped). Original header follows.
Auto-extracted from 3DAR_EV_all_in_one.ipynb cells 1-2 (unchanged code) on 2026-11-19.
Overrides: out_dir=runs/nb_binary_spec_20261119, tasks=[id]; id_task=binary, use_psd=False, use_rpm=False (notebook defaults).
Structure (senior memo): spectrogram -> CNN vector -> FCN -> latent -> decoder (Healthy/Fault)."""
# ==========================================
# 1. Configuration
# ==========================================
import csv, json, time, random, hashlib
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from sklearn.metrics import (confusion_matrix, f1_score, balanced_accuracy_score,
                             roc_auc_score, average_precision_score)
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PKG = Path('/home/kos/Desktop/3DAR_EV/EV_3DAR_AI')

CONFIG = {
    # --- 경로 ---
    'data_root': '/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV',  # 측정_목록.csv 가 있는 폴더 (읽기 전용)
    'out_dir':   str(PKG / 'runs/nb_binary_spec_strict_fmin500_20261123'),          # 이 노트북의 모든 결과
    'cache_dir': str(PKG / 'runs/fmin500_cache_20261123/cache'),    # 특징 캐시. 키가 같으면 다시 계산하지 않음
    'psd_file':  str(PKG / 'runs/notebook/psd_features.npz'),          # 없으면 계산해서 저장
    'fullrot_file': str(PKG / 'runs/notebook/fullrot_features.npz'),  # 셀 3용, 없으면 계산
    # --- 실행 ---
    'train': True,            # False면 학습 없이 ckpt_dir의 체크포인트로 평가만
    'ckpt_dir': None,         # None이면 out_dir. 예: str(PKG/'runs/baseline') 로 기존 결과 재평가
    'seeds': [42, 43, 44],
    'init_offset': 0,         # 분할은 seed 그대로, 가중치 초기화/셔플 seed만 바꿀 때
    'tasks': ['id'],
    'ssl_target': 'fault',    # 'fault': 고장 소리만 찾기 (개수 0~3, 고장 위치 각도). 'all': 기존(3음원 전부, 정답 항상 같음)
    'device': 'cuda',
    'gpu_duty': 0.5,          # GPU 평균 사용률 목표(발열 감소). 0.5 → 약 50%, 학습 시간 약 2배. 1.0 = 제한 없음
    # --- 특징 (configs/ev.json 과 같음) ---
    'sample_rate': 48000, 'n_fft': 2048, 'hop_length': 480, 'n_mels': 64,
    'fmin': 500.0, 'fmax': 12000.0, 'db_floor': -120.0,
    'angular_bins': 360, 'ssl_context_seconds': 1.0, 'id_frames': 100,
    'gaussian_sigma_degrees': 5.0, 'minimum_peak_spacing_degrees': 30,
    'geometry_status': 'assumed_from_recording_setup_verify_before_absolute_DOA_claims',
    'geometry': {'FRONT': 180.0, 'REAR': 0.0, 'RIGHT': 270.0},   # 미확인 기하 가정
    'rois': {'FRONT': [135.0, 225.0], 'REAR': [315.0, 405.0], 'RIGHT': [225.0, 315.0]},
    'positions': ['FRONT', 'REAR', 'RIGHT'],
    'classes': ['Healthy', 'Imbalance', 'Modulation', 'Whine', 'Wind', 'PWM'],
    'split_mode': 'source_disjoint_strict',
    # --- ID 모델 추가 입력 (FCN 앞에 concat) ---
    'id_task': 'binary',      # 'binary': 위치별 정상/고장만 / 'type': 6종(정상+고장 5종) 분류
    'use_rpm': False,         # 추정 RPM (측정값 아님)
    'use_psd': False,         # 위치 구간별 Welch PSD 64대역
    'psd_bands': 64,
    # --- 학습 ---
    'batch_size': 16, 'epochs': 50, 'patience': 10,
    'learning_rate': 1e-4, 'weight_decay': 1e-5,
    'location_loss_weight': 5.0, 'fault_threshold': 0.5, 'workers': 0,
}
POS = CONFIG['positions']; CLASSES = CONFIG['classes']
BINARY = CONFIG['id_task'] == 'binary'
ID_CLASSES = ['Healthy', 'Fault'] if BINARY else CLASSES      # ID 모델 출력 클래스

def id_label(c):
    return int(c != 'Healthy') if BINARY else CLASSES.index(c)

# ==========================================
# 0. 보조 함수 (파일 읽기/쓰기, 해시)
# ==========================================
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''): h.update(part)
    return h.hexdigest()

def data_path(root, relative):
    """Windows 경로 구분자 정리, 데이터 루트 밖으로 못 나가게 막음."""
    root = Path(root).resolve(); relative = str(relative).replace('\\', '/')
    if relative.startswith('/') or ':' in relative or '..' in Path(relative).parts:
        raise ValueError('Expected a relative dataset path: ' + relative)
    path = (root / relative).resolve()
    if not path.is_relative_to(root): raise ValueError('Dataset path escaped root.')
    return path

def records_by_ids(records, ids):
    lookup = {r['uid']: r for r in records}
    return [lookup[u] for u in ids]

# ==========================================
# 2. Parsing Dataset (측정_목록.csv + 녹음별 info CSV)
# ==========================================
def interval(text):
    a, b = map(float, text.split('~'))
    if not np.isfinite([a, b]).all() or b < a: raise ValueError('Invalid interval: ' + text)
    return [a, b]

def discover(root):
    """3대 스피커 녹음만 읽는다(1대 녹음 200개 제외). 라벨/개수 불일치는 바로 멈춘다."""
    root = Path(root).resolve(); records = []
    for row in read_csv(root / '측정_목록.csv'):
        if int(row['스피커수']) != 3: continue
        paths = {k: str(row[c]).replace('\\', '/') for k, c in
                 [('wav', 'WAV'), ('info', '실험정보CSV'), ('angles', '각도CSV')]}
        for rel in paths.values():
            if not data_path(root, rel).is_file(): raise FileNotFoundError(rel)
        info = read_csv(data_path(root, paths['info']))
        if len(info) != 3 or {s['Speaker_Loc'] for s in info} != set(POS):
            raise ValueError('Expected exactly three unique positions: ' + paths['info'])
        src = {s['Speaker_Loc']: s for s in info}; wav_name = Path(paths['wav']).name
        for pos, s in src.items():
            if s['status'] != 'OK' or s['Saved_Filename'] != wav_name:
                raise ValueError('Non-successful or inconsistent capture: ' + paths['info'])
            if s['Sound_Class'] not in CLASSES: raise ValueError(s['Sound_Class'])
            if s['Sound_Class'] != row[{'FRONT': '앞', 'REAR': '뒤', 'RIGHT': '오른쪽'}[pos]]:
                raise ValueError('Index/source label mismatch.')
        fault_count = sum(s['Sound_Class'] != 'Healthy' for s in info)
        if fault_count != int(row['고장음위치수']): raise ValueError('Fault count mismatch.')
        duration = float(info[0]['rotation_time_sec'])
        if not 3.7 <= duration <= 4.3: raise ValueError('Expected encoder-gated ~4 s audio.')
        rec = dict(uid=row['측정ID'], **paths, band=row['추정RPM구간'], fault_count=fault_count,
                   source_count=3, previous_role=row['학습분할'], original_order=int(row['원본회차']),
                   duration=duration, classes=[src[p]['Sound_Class'] for p in POS],
                   source_files=[src[p]['Source_File'].replace('\\', '/') for p in POS],
                   rpm_intervals=[interval(src[p]['Estimated_RPM_Interval']) for p in POS],
                   event_time=info[0]['event_time'])
        rec['input_fingerprint'] = {k: dict(bytes=data_path(root, v).stat().st_size,
                                            sha256=digest(data_path(root, v))) for k, v in paths.items()}
        records.append(rec)
    records.sort(key=lambda r: r['uid'])
    if len({r['uid'] for r in records}) != len(records): raise ValueError('Duplicate recording UID.')
    return records

def make_split(records, seed):
    """고장 위치 수 × RPM 구간 층화, 녹음 단위 8:1:1 (closed-source recording holdout)."""
    groups = defaultdict(list)
    for r in records: groups[(r['fault_count'], r['band'])].append(r['uid'])
    rng = np.random.default_rng(seed)
    split = dict(seed=seed, mode='recording_holdout_closed_source', train=[], validation=[], test=[])
    for key in sorted(groups):
        ids = np.array(sorted(groups[key]), dtype=object); rng.shuffle(ids)
        n = len(ids); a = int(.8 * n); b = int(.1 * n)
        if min(a, b, n - a - b) < 1: raise ValueError('Insufficient stratum: ' + str(key))
        split['train'] += ids[:a].tolist(); split['validation'] += ids[a:a + b].tolist()
        split['test'] += ids[a + b:].tolist()
    for k in ['train', 'validation', 'test']: split[k].sort()
    sets = [set(split[k]) for k in ['train', 'validation', 'test']]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)): raise AssertionError('Recording leakage.')
    lookup = {r['uid']: r for r in records}
    src = {k: {s for u in split[k] for s in lookup[u]['source_files']} for k in ['train', 'validation', 'test']}
    split['source_file_overlap'] = {a + '__' + b: len(src[a] & src[b])
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
    split['unseen_source_generalization_claim_allowed'] = False
    return split

# ==========================================
# 3. Feature Extraction (Log-Mel, 각도 정렬, ROI 잘라내기, PSD)
# ==========================================
def mel_bank():
    c = CONFIG
    hz2mel = lambda x: 2595.0 * np.log10(1.0 + np.asarray(x) / 700.0)
    mel2hz = lambda x: 700.0 * (10.0 ** (np.asarray(x) / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(c['fmin']), hz2mel(c['fmax']), c['n_mels'] + 2))
    f = np.fft.rfftfreq(c['n_fft'], 1.0 / c['sample_rate'])
    l = (f[None] - pts[:-2, None]) / (pts[1:-1, None] - pts[:-2, None])
    r = (pts[2:, None] - f[None]) / (pts[2:, None] - pts[1:-1, None])
    w = np.maximum(0.0, np.minimum(l, r)); w /= np.maximum(w.sum(1, keepdims=True), 1e-12)
    return w.astype(np.float32)

def logmel(audio):
    c = CONFIG; n = c['n_fft']
    if len(audio) < n: audio = np.pad(audio, (0, n - len(audio)))
    _, times, power = signal.spectrogram(audio.astype(np.float64), fs=c['sample_rate'], window='hann',
        nperseg=n, noverlap=n - c['hop_length'], detrend=False, scaling='spectrum', mode='psd')
    db = 10.0 * np.log10(np.maximum(mel_bank() @ power, 10.0 ** (c['db_floor'] / 10.0)))
    return db.astype(np.float32), times

def angle_track(path, duration):
    """각도 CSV → (시간, 각도) 단조 증가 쌍. 불연속은 고치지 않고 멈춘다."""
    rows = read_csv(path)
    t = np.array([float(r['elapsed_sec']) for r in rows]); a = np.array([float(r['angle_deg']) for r in rows])
    keep = np.isfinite(t) & np.isfinite(a) & (t >= 0) & (t <= duration); t, a = t[keep], a[keep]
    if len(t) < 20 or np.any(np.diff(t) < 0): raise ValueError('Invalid angle timestamps.')
    t, u = np.unique(t, return_index=True); a = a[u]
    if np.min(a) < 0 or np.max(a) > 360: raise ValueError('Angles outside 0..360.')
    if np.any(np.diff(a) < -1.0) or np.any(np.diff(a) > 20.0):
        raise ValueError('Unacceptable recorded angle discontinuity; no silent repair.')
    inc = np.r_[True, a[1:] > np.maximum.accumulate(a[:-1])]; t, a = t[inc], a[inc]
    interior = (t > 0) & (t < duration) & (a > 0) & (a < 360)
    return np.r_[0.0, t[interior], duration], np.r_[0.0, a[interior], 360.0]

def circular_segments(start, end):
    """뒤 구간 315–405°는 315–360 / 0–45 두 조각으로 분리 (경계 넘어 이어 붙이지 않음)."""
    if not 0 <= start < 360 or not start < end <= start + 360: raise ValueError('Invalid angular ROI.')
    return [(start, end)] if end <= 360 else [(start, 360.0), (0.0, end - 360.0)]

def ssl_input(db, times, at, aa):
    """360칸 각도별 Log-Mel (중심 프레임 + 1초 문맥 평균) → (2*n_mels, 360)."""
    c = CONFIG
    centers = np.interp(np.arange(360), aa, at)
    central = np.stack([np.interp(centers, times, band) for band in db])
    ctx = float(c['ssl_context_seconds'])
    if ctx <= 0: return central.astype(np.float32)
    p = np.power(10.0, db.astype(np.float64) / 10.0)
    sums = np.c_[np.zeros(len(db)), np.cumsum(p, axis=1)]
    lo = np.searchsorted(times, centers - ctx / 2, 'left'); hi = np.searchsorted(times, centers + ctx / 2, 'right')
    lo = np.minimum(lo, len(times) - 1); hi = np.maximum(hi, lo + 1)
    local = 10 * np.log10(np.maximum((sums[:, hi] - sums[:, lo]) / np.maximum(hi - lo, 1), 10 ** (c['db_floor'] / 10)))
    return np.concatenate([central, local], axis=0).astype(np.float32)

def extract_record(rec):
    c = CONFIG; root = c['data_root']
    audio, fs = sf.read(data_path(root, rec['wav']), dtype='float32')
    if fs != c['sample_rate'] or audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError('Expected finite mono 48 kHz WAV: ' + rec['uid'])
    duration = len(audio) / fs
    if abs(duration - rec['duration']) > .002: raise ValueError('WAV/metadata duration mismatch.')
    at, aa = angle_track(data_path(root, rec['angles']), duration)
    db, times = logmel(audio)
    m, T = c['n_mels'], c['id_frames']
    views = np.full((3, 2, m, T), c['db_floor'], np.float32); masks = np.zeros((3, 2, T), np.float32)
    durs = np.zeros((3, 2), np.float32); rpm = np.zeros((3, 2), np.float32)
    common_rpm = np.median(np.array(rec['rpm_intervals']), axis=0)
    for pi, pos in enumerate(POS):
        mids, wts = [], []
        for vi, (a, b) in enumerate(circular_segments(*c['rois'][pos])):
            t0, t1 = np.interp([a, b], aa, at)
            seg = audio[round(t0 * fs):round(t1 * fs)]
            if len(seg) < 2: raise ValueError('Empty angular ROI.')
            part, _ = logmel(seg); n = part.shape[1]
            if n > T:   # 특징의 시간축만 리샘플 (원음/라벨/RPM은 그대로)
                part = np.stack([np.interp(np.linspace(0, n - 1, T), np.arange(n), v) for v in part]); n = T
            views[pi, vi, :, :n] = part; masks[pi, vi, :n] = 1; durs[pi, vi] = t1 - t0
            mids.append(np.interp((t0 + t1) / 2, [0, duration], common_rpm)); wts.append(t1 - t0)
        mean = np.average(mids, weights=wts); spread = np.sqrt(np.average((np.asarray(mids) - mean) ** 2, weights=wts))
        rpm[pi] = [mean / 10000.0, spread / 10000.0]
    return dict(ssl=ssl_input(db, times, at, aa), id_views=views, id_masks=masks, view_durations=durs, rpm=rpm)

FEATURE_KEYS = ['sample_rate', 'n_fft', 'hop_length', 'n_mels', 'fmin', 'fmax', 'db_floor',
                'angular_bins', 'ssl_context_seconds', 'id_frames', 'positions', 'rois']

def feature_key(rec):
    """ev3dar.features.feature_key 와 같은 계산 → 기존 runs/baseline/cache 재사용 가능."""
    cfg = {k: CONFIG[k] for k in FEATURE_KEYS}
    return hashlib.sha256(json.dumps(dict(version=1, uid=rec['uid'], cfg=cfg,
        inputs=rec['input_fingerprint']), sort_keys=True).encode()).hexdigest()

def cache_path(uid):
    if not uid.replace('_', '').isalnum(): raise ValueError('Invalid cache UID.')
    return Path(CONFIG['cache_dir']) / (uid + '.npz')

def build_cache(records):
    Path(CONFIG['cache_dir']).mkdir(parents=True, exist_ok=True); made = 0
    for i, r in enumerate(records, 1):
        p = cache_path(r['uid']); key = feature_key(r)
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                if str(z['feature_key'].item()) == key: continue
        tmp = p.with_suffix('.pending.npz')
        np.savez_compressed(tmp, feature_key=np.array(key), **extract_record(r)); tmp.replace(p); made += 1
        if made % 25 == 0: print(f'Features {i}/{len(records)}', flush=True)
    print(f'Feature cache ready: {len(records)} recordings ({made} newly computed)')

PSD_EDGES = np.geomspace(20, 20000, CONFIG['psd_bands'] + 1)

def psd_vec(seg, fs):
    f, p = signal.welch(seg, fs=fs, window='hann', nperseg=min(4096, len(seg)), noverlap=None)
    out = np.empty(len(PSD_EDGES) - 1)
    for i in range(len(out)):
        m = (f >= PSD_EDGES[i]) & (f < PSD_EDGES[i + 1])
        out[i] = p[m].mean() if m.any() else np.interp(np.sqrt(PSD_EDGES[i] * PSD_EDGES[i + 1]), f, p)
    return out

def extract_psd(rec):
    """위치 구간별 Welch PSD → 로그 간격 64대역(20 Hz–20 kHz), 뒤 두 조각은 길이 가중 평균 → dB (3, 64)."""
    x, fs = sf.read(data_path(CONFIG['data_root'], rec['wav']), dtype='float64')
    at, aa = angle_track(data_path(CONFIG['data_root'], rec['angles']), len(x) / fs)
    res = np.zeros((3, len(PSD_EDGES) - 1))
    for pi, pos in enumerate(POS):
        acc = 0; wsum = 0
        for a, b in circular_segments(*CONFIG['rois'][pos]):
            t0, t1 = np.interp([a, b], aa, at)
            acc = acc + psd_vec(x[round(t0 * fs):round(t1 * fs)], fs) * (t1 - t0); wsum += t1 - t0
        res[pi] = 10 * np.log10(acc / wsum + 1e-20)
    return res.astype(np.float32)

def load_or_build_psd(records):
    path = Path(CONFIG['psd_file'])
    if path.exists():
        z = np.load(path); by = {str(u): z['psd'][i] for i, u in enumerate(z['uids'])}
        if all(r['uid'] in by for r in records):
            print('PSD loaded:', path); return by
    print('Computing PSD for', len(records), 'recordings ...', flush=True)
    P = np.stack([extract_psd(r) for r in records])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, psd=P, uids=np.array([r['uid'] for r in records]), band_edges_hz=PSD_EDGES)
    return {r['uid']: P[i] for i, r in enumerate(records)}

# ==========================================
# 4. Normalization & Dataset (train 녹음 통계만 사용)
# ==========================================
def fit_normalization(train_records, psd_by=None):
    ts = tq = None; tn = 0; is_ = iq = None; inn = 0
    for r in train_records:
        with np.load(cache_path(r['uid']), allow_pickle=False) as z:
            if str(z['feature_key'].item()) != feature_key(r): raise ValueError('Stale cache.')
            x = z['ssl'].astype(np.float64)
            if ts is None: ts = np.zeros(x.shape[0]); tq = ts.copy()
            ts += x.sum(1); tq += (x * x).sum(1); tn += x.shape[1]
            a = z['id_views'].astype(np.float64); m = z['id_masks'][:, :, None, :]
            if is_ is None: is_ = np.zeros(a.shape[2]); iq = is_.copy()
            is_ += (a * m).sum((0, 1, 3)); iq += (a * a * m).sum((0, 1, 3)); inn += m.sum()
    def stats(s, q, n):
        mean = s / n; return dict(mean=mean.tolist(), std=np.sqrt(np.maximum(q / n - mean * mean, 1.0)).tolist())
    norm = dict(ssl=stats(ts, tq, tn), id=stats(is_, iq, inn),
                fitted_recording_ids=[r['uid'] for r in train_records], train_only=True)
    if psd_by is not None:
        S = np.stack([psd_by[r['uid']] for r in train_records])
        norm['psd'] = dict(mean=S.mean((0, 1)).tolist(), std=np.maximum(S.std((0, 1)), 1e-3).tolist())
    return norm

def ssl_angles(rec, mode):
    """SSL 정답 각도. 'fault' = 고장 소리가 난 위치만 (Healthy는 어디서나 나는 배경이라 제외)."""
    return [CONFIG['geometry'][p] for p, c in zip(POS, rec['classes']) if mode != 'fault' or c != 'Healthy']

def ssl_count_label(rec, mode):
    return rec['fault_count'] if mode == 'fault' else rec['source_count'] - 1   # fault: 0~3 / all: 1~3 → 0~2

def gaussian_target(degrees, sigma=5.0):
    x = np.arange(360); target = np.zeros(360, np.float32)
    for ang in degrees:
        d = np.abs(x - (float(ang) % 360)); d = np.minimum(d, 360 - d)
        target = np.maximum(target, np.exp(-.5 * (d / sigma) ** 2))
    return target

class EVDataset(Dataset):
    def __init__(self, records, norm, psd_by=None):
        self.records, self.psd_by = records, psd_by
        self.sm = np.asarray(norm['ssl']['mean'], np.float32)[:, None]
        self.ss = np.asarray(norm['ssl']['std'], np.float32)[:, None]
        self.im = np.asarray(norm['id']['mean'], np.float32)[None, None, :, None]
        self.istd = np.asarray(norm['id']['std'], np.float32)[None, None, :, None]
        self.psd_norm = norm.get('psd')
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        r = self.records[i]
        with np.load(cache_path(r['uid']), allow_pickle=False) as z:
            if str(z['feature_key'].item()) != feature_key(r): raise ValueError('Stale feature cache.')
            ssl = z['ssl'].astype(np.float32); views = z['id_views'].astype(np.float32)
            masks = z['id_masks'].copy(); durs = z['view_durations'].copy(); rpm = z['rpm'].copy()
        ssl = (ssl - self.sm) / self.ss; views = ((views - self.im) / self.istd) * masks[:, :, None, :]
        if self.psd_by is not None and self.psd_norm is not None:
            psd = (self.psd_by[r['uid']] - np.asarray(self.psd_norm['mean'], np.float32)) / \
                  np.asarray(self.psd_norm['std'], np.float32)
        else:
            psd = np.zeros((3, 1), np.float32)
        return dict(uid=r['uid'], ssl=torch.from_numpy(ssl), views=torch.from_numpy(views),
                    masks=torch.from_numpy(masks), durations=torch.from_numpy(durs), rpm=torch.from_numpy(rpm),
                    psd=torch.from_numpy(psd.astype(np.float32)),
                    labels=torch.tensor([id_label(c) for c in r['classes']], dtype=torch.long),
                    count=torch.tensor(ssl_count_label(r, CONFIG['ssl_target']), dtype=torch.long),
                    target=torch.from_numpy(gaussian_target(ssl_angles(r, CONFIG['ssl_target']), CONFIG['gaussian_sigma_degrees'])))

# ==========================================
# 5. Model Architecture & Loss Functions
# ==========================================
class LocalizationCNN(nn.Module):
    """SSL: 펭귄식 원형 1D CNN → 음원 수(1~3) + 360칸 방향 점수."""
    def __init__(self, channels, n_count=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 128, 7, padding=3, padding_mode='circular'), nn.InstanceNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 5, padding=2, padding_mode='circular'), nn.InstanceNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, 5, padding=4, dilation=2, padding_mode='circular'), nn.InstanceNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding=1, padding_mode='circular'), nn.InstanceNorm1d(128), nn.ReLU())
        self.count_branch = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 64),
                                          nn.ReLU(), nn.Dropout(.3), nn.Linear(64, n_count))
        self.loc_branch = nn.Sequential(nn.Conv1d(128, 64, 3, padding=1, padding_mode='circular'),
                                        nn.ReLU(), nn.Conv1d(64, 1, 1))
    def forward(self, x):
        f = self.features(x); return self.count_branch(f), self.loc_branch(f).squeeze(1)

class SoundTypeCNN(nn.Module):
    """ID: 스펙트로그램 2D CNN → 벡터 (+PSD/RPM concat) → FCN → 64차원 latent → 6종 분류 decoder."""
    def __init__(self, classes=6, use_rpm=False, psd_dim=0):
        super().__init__(); self.use_rpm, self.psd_dim = use_rpm, psd_dim
        layers = []; prev = 1
        for c in [32, 64, 128]:
            layers += [nn.Conv2d(prev, c, 3, padding=1), nn.GroupNorm(8, c), nn.GELU(), nn.MaxPool2d(2)]; prev = c
        self.encoder = nn.Sequential(*layers)
        self.fcn = nn.Sequential(nn.Linear(128 + psd_dim + (2 if use_rpm else 0), 128), nn.GELU(), nn.Dropout(.2),
                                 nn.Linear(128, 64), nn.GELU())
        self.decoder = nn.Linear(64, classes)   # 분류 decoder (파형 복원 아님)
    def forward(self, views, masks, durations, rpm, psd=None):
        b, p, v, m, t = views.shape
        x = self.encoder(views.reshape(b * p * v, 1, m, t))
        valid = F.adaptive_avg_pool1d(masks.reshape(b * p * v, 1, t), x.shape[-1]).unsqueeze(2)
        z = (x * valid).sum((2, 3)) / (valid.sum((2, 3)) * x.shape[2]).clamp_min(1e-6)
        z = z.reshape(b, p, v, -1)
        w = durations * (masks.sum(-1) > 0).to(durations.dtype)          # 뒤 두 조각: 길이 가중 평균
        z = (z * w.unsqueeze(-1)).sum(2) / w.sum(2, keepdim=True).clamp_min(1e-6)
        if self.psd_dim: z = torch.cat([z, psd], dim=-1)
        if self.use_rpm: z = torch.cat([z, rpm], dim=-1)
        latent = self.fcn(z)
        return self.decoder(latent), latent

def focal_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return (.25 * (1 - torch.exp(-bce)).pow(2) * bce).mean()

def build_model(task, cfg):
    if task == 'ssl': return LocalizationCNN(cfg['n_mels'] * (2 if cfg['ssl_context_seconds'] > 0 else 1),
                                             4 if cfg.get('ssl_target', 'all') == 'fault' else 3)
    return SoundTypeCNN(2 if cfg.get('id_task', 'type') == 'binary' else len(cfg['classes']), cfg.get('use_rpm', False),
                        cfg.get('psd_bands', 64) if cfg.get('use_psd', False) else 0)

# ==========================================
# 6. Evaluation Functions
# ==========================================
def circular_peaks(prob, count, min_spacing=30):
    prob = np.asarray(prob, float)
    if count <= 0: return []                                         # 고장 0곳이면 각도 없음
    peaks, _ = find_peaks(np.tile(prob, 3), prominence=.01)          # 0/359°도 내부 후보로
    cand = sorted([int(i - 360) for i in peaks if 360 <= i < 720], key=lambda i: (-prob[i], i)); sel = []
    for c in cand:
        if all(min(abs(c - p), 360 - abs(c - p)) >= min_spacing for p in sel): sel.append(c)
        if len(sel) >= count: break
    return sel

def angular_error(pred, truth):
    truth = np.asarray(truth, float); pred = np.asarray(pred, float)
    if len(truth) == 0: return 0.0
    if len(pred) == 0: return 180.0
    cost = np.abs(truth[:, None] - pred[None]); cost = np.minimum(cost, 360 - cost)
    i, j = linear_sum_assignment(cost)
    return float((cost[i, j].sum() + 180.0 * (len(truth) - len(i))) / len(truth))   # 못 찾은 음원은 180°

def classification_metrics(labels, probs, threshold=.5):
    labels = np.asarray(labels, int).reshape(-1); probs = np.asarray(probs, float).reshape(-1, len(ID_CLASSES))
    pred = probs.argmax(-1); fault = (labels != 0).astype(int); score = 1 - probs[:, 0]
    binary = (score >= threshold).astype(int); both = len(np.unique(fault)) == 2
    return dict(n_regions=len(labels), type_accuracy=float(np.mean(pred == labels)),
        type_macro_f1=float(f1_score(labels, pred, labels=np.arange(len(ID_CLASSES)), average='macro', zero_division=0)),
        type_confusion_matrix=confusion_matrix(labels, pred, labels=np.arange(len(ID_CLASSES))).tolist(),
        fault_accuracy=float(np.mean(binary == fault)), fault_f1=float(f1_score(fault, binary, zero_division=0)),
        fault_balanced_accuracy=float(balanced_accuracy_score(fault, binary)) if both else float(np.mean(binary == fault)),
        fault_roc_auc=float(roc_auc_score(fault, score)) if both else None,
        fault_average_precision=float(average_precision_score(fault, score)) if both else None,
        fault_confusion_matrix=confusion_matrix(fault, binary, labels=[0, 1]).tolist())

# ==========================================
# 7. Train Loop
# ==========================================
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

class GpuThrottle:
    """학습 step마다 GPU가 일한 시간에 비례해 쉬어서 평균 사용률을 gpu_duty 근처로 낮춘다.
    결과(가중치/지표)는 같고 시간만 늘어난다. GB10은 nvidia-smi 전력 제한이 안 돼서 소프트웨어로 한다."""
    def __init__(self, duty): self.duty = float(duty); self.t = time.perf_counter()
    def reset(self): self.t = time.perf_counter()
    def step(self):
        if self.duty >= 1: return
        if torch.cuda.is_available(): torch.cuda.synchronize()
        busy = time.perf_counter() - self.t
        time.sleep(busy * (1 - self.duty) / max(self.duty, 0.05)); self.t = time.perf_counter()

def make_loader(records, norm, psd_by, shuffle=False, seed=42):
    return DataLoader(EVDataset(records, norm, psd_by), batch_size=CONFIG['batch_size'], shuffle=shuffle,
                      num_workers=CONFIG['workers'], generator=torch.Generator().manual_seed(seed),
                      pin_memory=torch.cuda.is_available())

def forward(model, task, batch, device):
    if task == 'ssl': return model(batch['ssl'].to(device, non_blocking=True))
    return model(*(batch[k].to(device, non_blocking=True) for k in ['views', 'masks', 'durations', 'rpm', 'psd']))

def ssl_loss(counts, loc, batch, device):
    return F.cross_entropy(counts, batch['count'].to(device)) + \
           CONFIG['location_loss_weight'] * focal_loss(loc, batch['target'].to(device))

def get_device():
    if CONFIG['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run on the Spark GPU; no automatic CPU fallback.')
    return torch.device(CONFIG['device'])

def validate(model, task, loader, device):
    model.eval(); loss_sum = 0.0; Y = []; P = []
    with torch.no_grad():
        for b in loader:
            a, c = forward(model, task, b, device)
            if task == 'ssl': loss_sum += float(ssl_loss(a, c, b, device)) * len(b['uid'])
            else: Y.append(b['labels'].numpy()); P.append(a.softmax(-1).cpu().numpy())
    if task == 'ssl':
        loss = loss_sum / len(loader.dataset); return dict(loss=loss), -loss
    m = classification_metrics(np.concatenate(Y), np.concatenate(P), CONFIG['fault_threshold'])
    return m, m['type_macro_f1']            # ID는 validation macro-F1로 best 선택

def train_task(task, seed, split, train_recs, valid_recs, norm, psd_by, run_dir):
    device = get_device(); seed_all(seed + CONFIG['init_offset'])
    tr = make_loader(train_recs, norm, psd_by, True, seed + CONFIG['init_offset'])
    va = make_loader(valid_recs, norm, psd_by)
    model = build_model(task, CONFIG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    best = -float('inf'); stale = 0; history = []; throttle = GpuThrottle(CONFIG['gpu_duty'])
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train(); total = 0.0; n = 0; tick = time.monotonic(); throttle.reset()
        for b in tr:
            opt.zero_grad(set_to_none=True); a, c = forward(model, task, b, device)
            loss = ssl_loss(a, c, b, device) if task == 'ssl' else \
                   F.cross_entropy(a.reshape(-1, len(ID_CLASSES)), b['labels'].to(device).reshape(-1))
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss.')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach()) * len(b['uid']); n += len(b['uid']); throttle.step()
        metrics, score = validate(model, task, va, device)
        history.append(dict(epoch=epoch, train_loss=total / n, selection_score=score,
                            validation={k: v for k, v in metrics.items() if 'matrix' not in k},
                            seconds=time.monotonic() - tick))
        print(f"[seed{seed} {task}] epoch {epoch:3d}  train_loss {total / n:.4f}  "
              f"val {'loss' if task == 'ssl' else 'macroF1'} {abs(score):.4f}", flush=True)
        if score > best:
            best = score; stale = 0
            ckpt = dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        task=task, config=dict(CONFIG), normalization=norm, seed=seed, epoch=epoch,
                        validation=metrics, split=split, training_claim='closed-source held-out-recording baseline')
            tmp = run_dir / f'{task}_best.pending.pt'; torch.save(ckpt, tmp); tmp.replace(run_dir / f'{task}_best.pt')
        else:
            stale += 1
        write_json(run_dir / f'{task}_history.json', history)
        if stale >= CONFIG['patience']: break
    print(f'[seed{seed} {task}] best epoch {max(history, key=lambda h: h["selection_score"])["epoch"]}', flush=True)
    del model, opt
    if device.type == 'cuda': torch.cuda.empty_cache()

# ==========================================
# 8. Multi-Seed Loop Start!
# ==========================================
OUT = Path(CONFIG['out_dir']); OUT.mkdir(parents=True, exist_ok=True)
print('Reading dataset (read-only):', CONFIG['data_root'], flush=True)
RECORDS = discover(CONFIG['data_root'])
_groups = Counter(r['fault_count'] for r in RECORDS)
if len(RECORDS) != 1200 or _groups != {0: 300, 1: 300, 2: 300, 3: 300}:
    raise ValueError('Expected 1200 three-speaker recordings (300 per fault count): ' + str(_groups))
LOOKUP = {r['uid']: r for r in RECORDS}
SPLITS = {s: read_json(PKG / f'runs/strict_split_20261123/seed{s}.json') for s in CONFIG['seeds']}
(OUT / 'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in RECORDS), encoding='utf-8')
for s, sp in SPLITS.items(): write_json(OUT / f'splits/seed{s}.json', sp)
write_json(OUT / 'config.json', CONFIG)
print('Recordings:', len(RECORDS), dict(_groups), '| split sizes:',
      {k: len(SPLITS[CONFIG['seeds'][0]][k]) for k in ['train', 'validation', 'test']})

build_cache(RECORDS)
PSD_BY = load_or_build_psd(RECORDS) if CONFIG['use_psd'] else None

if CONFIG['train']:
    for seed in CONFIG['seeds']:
        sp = SPLITS[seed]; run_dir = OUT / f'seed{seed}'; run_dir.mkdir(exist_ok=True)
        train_recs = records_by_ids(RECORDS, sp['train']); valid_recs = records_by_ids(RECORDS, sp['validation'])
        norm = fit_normalization(train_recs, PSD_BY); write_json(run_dir / 'normalization.json', norm)
        for task in CONFIG['tasks']:
            train_task(task, seed, sp, train_recs, valid_recs, norm, PSD_BY, run_dir)
    print('Training done. Raw dataset was read only.')

# ==========================================
# 1. Test Evaluation (seed별 → test_predictions.csv / test_metrics.json)
# ==========================================
def load_ckpt(path, task, device):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)   # 이 프로젝트가 만든 파일만 로드
    if ckpt['task'] != task: raise ValueError('Checkpoint task mismatch.')
    model = build_model(task, ckpt['config']); model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval(), ckpt

def evaluate(seed):
    device = get_device(); ck_dir = Path(CONFIG['ckpt_dir'] or CONFIG['out_dir']) / f'seed{seed}'
    out_dir = OUT / f'seed{seed}'; out_dir.mkdir(parents=True, exist_ok=True)
    id_model, ic = load_ckpt(ck_dir / 'id_best.pt', 'id', device)
    has_ssl = (ck_dir / 'ssl_best.pt').exists()
    if has_ssl:
        ssl_model, sc = load_ckpt(ck_dir / 'ssl_best.pt', 'ssl', device)
        if sc['split']['test'] != ic['split']['test']: raise ValueError('SSL/ID split mismatch.')
    if ic['split']['test'] != SPLITS[seed]['test']: raise ValueError('Checkpoint split differs from this notebook split.')
    test = records_by_ids(RECORDS, ic['split']['test'])
    psd_by = PSD_BY if ic['config'].get('use_psd') else None
    if ic['config'].get('use_psd') and psd_by is None: psd_by = load_or_build_psd(RECORDS)
    loader = make_loader(test, ic['normalization'], psd_by)
    mode = sc['config'].get('ssl_target', 'all') if has_ssl else None   # 기존 체크포인트는 'all'
    rows = []; Y = []; P = []; maes = []; count_ok = 0; pos_exact = defaultdict(list)
    with torch.no_grad():
        for b in loader:
            ip = forward(id_model, 'id', b, device)[0].softmax(-1).cpu().numpy()
            yy = b['labels'].numpy(); Y.append(yy); P.append(ip)
            if has_ssl:
                cnt, loc = forward(ssl_model, 'ssl', b, device)
                dp = loc.sigmoid().cpu().numpy(); predn = cnt.argmax(-1).cpu().numpy() + (0 if mode == 'fault' else 1)
            for j, uid in enumerate(b['uid']):
                peaks = []
                if has_ssl:
                    peaks = circular_peaks(dp[j], int(predn[j]), CONFIG['minimum_peak_spacing_degrees'])
                    truth = ssl_angles(LOOKUP[uid], mode)
                    if truth: maes.append(angular_error(peaks, truth))       # 고장 0곳 녹음은 각도 오차 없음
                    count_ok += int(predn[j] == len(truth))
                ssl_pos = []
                for k, pos in enumerate(POS):
                    a, bb = CONFIG['rois'][pos]
                    inside = [p for p in peaks if (a <= p < bb if bb <= 360 else p >= a or p < bb - 360)]
                    ssl_pos.append(int(bool(inside)))
                    fp = float(1 - ip[j, k, 0])
                    rows.append(dict(recording_id=uid, seed=seed, position=pos, fault_count=LOOKUP[uid]['fault_count'],
                        predicted_type=ID_CLASSES[int(ip[j, k].argmax())], gt_type=LOOKUP[uid]['classes'][k],
                        predicted_fault=int(fp >= CONFIG['fault_threshold']), gt_fault=int(yy[j, k] != 0),
                        fault_probability=fp,
                        predicted_angle_deg=max(inside, key=lambda p: dp[j, p]) if inside else None,
                        predicted_source_count=int(predn[j]) if has_ssl else None,
                        ssl_predicted_fault=int(bool(inside)) if has_ssl and mode == 'fault' else None,
                        band=LOOKUP[uid]['band'], **{'prob_' + c: float(ip[j, k, ci]) for ci, c in enumerate(ID_CLASSES)}))
                if has_ssl and mode == 'fault':   # 세 위치(앞/뒤/오른쪽)의 고장 여부를 SSL이 모두 맞혔나
                    gt = [int(c != 'Healthy') for c in LOOKUP[uid]['classes']]
                    pos_exact[LOOKUP[uid]['fault_count']].append(ssl_pos == gt)
    Yc = np.concatenate(Y); Pc = np.concatenate(P)
    m = classification_metrics(Yc, Pc, CONFIG['fault_threshold'])
    # train 분포만으로 맞히는 기준선 (소리를 안 듣고 위치별 클래스 비율로 예측)
    prior = np.ones((3, len(ID_CLASSES)))
    for r in records_by_ids(RECORDS, ic['split']['train']):
        for k, c in enumerate(r['classes']): prior[k, id_label(c)] += 1
    prior /= prior.sum(-1, keepdims=True)
    m['training_position_prior_baseline'] = classification_metrics(Yc, np.broadcast_to(prior, Pc.shape), CONFIG['fault_threshold'])
    m['per_position'] = {p: classification_metrics(Yc[:, k], Pc[:, k], CONFIG['fault_threshold']) for k, p in enumerate(POS)}
    if has_ssl:
        m.update(ssl_target=mode, ssl_count_accuracy=count_ok / len(test), ssl_mae_degrees=float(np.mean(maes)))
        if mode == 'fault':
            m['ssl_position_exact'] = float(np.mean([v for vs in pos_exact.values() for v in vs]))
            m.update({f'ssl_position_exact_{k}fault': float(np.mean(v)) for k, v in sorted(pos_exact.items())})
        else:
            m['ssl_labels_constant'] = True   # 모든 녹음이 3음원·같은 배치 → SSL 정답이 상수
    m.update(seed=seed, n_recordings=len(test), id_best_epoch=ic['epoch'],
             source_file_overlap=ic['split']['source_file_overlap'],
             generalization_claim='Known replay-source recordings only; NOT unseen-source localization.')
    write_json(out_dir / 'test_metrics.json', m); write_csv(out_dir / 'test_predictions.csv', rows)
    return m, rows

# ==========================================
# 2. 위치 판별 지표 (11/16 compare.py 와 같은 정의)
# ==========================================
def location_metrics(rows):
    """exact1: 고장 1곳 녹음에서 세 위치의 정상/고장 판정이 모두 맞은 비율.
       other1: 고장 1곳 녹음에서 정상 위치를 고장으로 오경보한 녹음 비율.
       FA{k}: 고장 k곳 녹음의 정상 위치 오경보율. miss: 고장 위치를 정상으로 놓친 비율.
       (RESULTS.md 표의 '고장 1곳: 정상 위치 오경보'는 FA1 이다.)"""
    by = defaultdict(list)
    for r in rows: by[r['recording_id']].append(r)
    out = {}
    for k in range(3):
        h = [r for v in by.values() if v[0]['fault_count'] == k for r in v if not r['gt_fault']]
        out[f'FA{k}'] = float(np.mean([r['predicted_fault'] for r in h])) if h else float('nan')
    one = [v for v in by.values() if v[0]['fault_count'] == 1]
    out['exact1'] = float(np.mean([all(r['predicted_fault'] == r['gt_fault'] for r in v) for v in one]))
    out['other1'] = float(np.mean([any(r['predicted_fault'] and not r['gt_fault'] for r in v) for v in one]))
    out['miss'] = float(np.mean([1 - r['predicted_fault'] for r in rows if r['gt_fault']]))
    for c in ([] if BINARY else CLASSES[1:]):
        R = [r for r in rows if r['gt_type'] == c]; out['recall_' + c] = float(np.mean([r['predicted_type'] == c for r in R]))
    return out

# ==========================================
# 3. Multi-Seed Evaluation & Grand Summary (Mean ± Std)
# ==========================================
ALL_METRICS, ALL_ROWS, LOC_METRICS = {}, [], {}
for seed in CONFIG['seeds']:
    m, rows = evaluate(seed); ALL_METRICS[seed] = m; ALL_ROWS += rows; LOC_METRICS[seed] = location_metrics(rows)

KEYS = ['type_accuracy', 'type_macro_f1', 'fault_f1', 'fault_balanced_accuracy', 'fault_roc_auc',
        'ssl_count_accuracy', 'ssl_mae_degrees', 'ssl_position_exact'] + [f'ssl_position_exact_{k}fault' for k in range(4)]
LOC_KEYS = ['exact1', 'other1', 'miss', 'FA0', 'FA1', 'FA2'] + ['recall_' + c for c in ([] if BINARY else CLASSES[1:])]
SUMMARY = {}
print(f"{'metric':34s}" + ''.join(f'seed{s:<8d}' for s in CONFIG['seeds']) + 'mean ± std')
for k in KEYS + LOC_KEYS:
    vals = [(LOC_METRICS[s] if k in LOC_KEYS else ALL_METRICS[s]).get(k) for s in CONFIG['seeds']]
    if k == 'ssl_mae_degrees' and ALL_METRICS[CONFIG['seeds'][0]].get('ssl_target') == 'fault': k = 'ssl_fault_mae_degrees'
    if any(v is None for v in vals): continue
    mean = float(np.mean(vals)); std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    SUMMARY[k] = dict(mean=mean, std=std, per_seed=vals)
    print(f'{k:34s}' + ''.join(f'{v:<12.3f}' for v in vals) + f'{mean:.3f} ± {std:.3f}')
prior_f1 = np.mean([ALL_METRICS[s]['training_position_prior_baseline']['type_macro_f1'] for s in CONFIG['seeds']])
print(f'\n(참고) 소리 없이 위치별 train 분포로만 맞힌 macroF1: {prior_f1:.3f}')
write_json(OUT / 'three_seed_summary.json', dict(seeds=CONFIG['seeds'], protocol=CONFIG['split_mode'],
    summary=SUMMARY, config=CONFIG,
    warning='No unseen-source generalization claim (closed-source recording holdout).'))