"""Analysis of the ORIGINAL KU Leuven source files (2200 wav: Healthy 1200, 5 fault classes x 200), 2026-11-23.
Files: /home/kos/Desktop/3DAR_EV/KU_Leuven_original/<Class>/*.wav (downloaded by the user, read-only here).

Per file:
  1. RPM profile fit. Paper: run-up 0 -> 10,000 RPM over the file, mid-point perturbed <= 10 %. Model
     rpm(t) = 10000 (u + c u (1-u)), u = t/T; c chosen to maximise the sharpness of the healthy orders 4/8/12/16
     (audio decimated to 8 kHz, order tracking 64 samples/rev, window u in [0.35, 0.9]).
  2. Features in the window 5000 <= rpm <= 8000 (the range played in the 3DAR experiment):
     band levels (dBFS, Welch) and fault-signature features (order-tracked, same definitions as
     experiments/knowledge_rules_20261123): order 1/2 (Imbalance), order 7.9/8.1 sidebands (Modulation),
     order 16 vs orders 8/12 (Whine), 5 kHz line (PWM), 3.75-5.75 kHz band vs neighbours (Wind).
Part B (paired): each 1-speaker recording vs the exact source segment that was played (info CSV interval):
  band transfer (recorded - original, dB) and how much of each fault signature survives playback + 3DAR recording.
Outputs: runs/ku_original_20261123/{file_features.json, paired_1spk.json, summary.json}, figures/ku_original_summary.png
"""
from pathlib import Path
import sys, json, csv, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf
from scipy import signal

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT / 'experiments/knowledge_rules_20261123'))
from ev3dar.common import read_json, write_json, data_path
import knowledge_detector as K

KU = Path('/home/kos/Desktop/3DAR_EV/KU_Leuven_original')
ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
OUT = PROJECT / 'runs/ku_original_20261123'
CLASSES = ['Healthy', 'Imbalance', 'Modulation', 'Whine', 'PWM', 'Wind']
EDGES = [100, 300, 700, 1500, 2500, 3750, 5750, 8000, 12000, 20000]
BN = [f'{a}-{b}' for a, b in zip(EDGES[:-1], EDGES[1:])]


def rpm_model(c, T): return lambda t: 10000.0 * ((lambda u: u + c * u * (1 - u))(np.clip(np.asarray(t) / T, 0, 1)))


def fit_c(audio, fs):
    T = len(audio) / fs; x = signal.resample_poly(audio.astype(np.float64), 1, 6); f8 = fs / 6
    i0, i1 = int(.35 * len(x)), int(.9 * len(x)); seg = x[i0:i1]; t = (i0 + np.arange(len(seg))) / f8
    def sharp(c):
        rev = np.concatenate([[0], np.cumsum(rpm_model(c, T)(t[:-1]) / 60 / f8)]); m = int(rev[-1] * 64)
        xr = np.interp(np.arange(m) / 64, rev, seg); P = 10 * np.log10(np.abs(np.fft.rfft(xr * np.hanning(m))) ** 2 + 1e-30)
        o = np.fft.rfftfreq(m, 1 / 64); return np.mean([K.peak_over_bg(o, P, k, half=0.02) for k in (4, 8, 12, 16)])
    grid = np.arange(-0.30, 0.301, 0.03); s = [sharp(c) for c in grid]; c0 = grid[int(np.argmax(s))]
    fine = np.arange(c0 - 0.03, c0 + 0.0301, 0.005); s2 = [sharp(c) for c in fine]
    return float(fine[int(np.argmax(s2))]), float(max(s2)), float(sharp(0.0))


def features(seg, t0, fs, rpm_fn):
    o, P, _ = K.order_spectrum(seg, t0, fs, rpm_fn)
    raw = lambda c: float(P[(o >= c - .03) & (o <= c + .03)].max())
    f = dict(imb_o1=K.peak_over_bg(o, P, 1.0), imb_o2=K.peak_over_bg(o, P, 2.0),
             mod_sb=max(K.peak_over_bg(o, P, 7.9, exclude=(8.0, 8.1)), K.peak_over_bg(o, P, 8.1, exclude=(8.0, 7.9))),
             o4=K.peak_over_bg(o, P, 4.0), o8=K.peak_over_bg(o, P, 8.0), o12=K.peak_over_bg(o, P, 12.0), o16=K.peak_over_bg(o, P, 16.0),
             whine_ratio=raw(16.0) - 0.5 * (raw(8.0) + raw(12.0)))
    fr, pw = signal.welch(seg.astype(np.float64), fs, nperseg=min(16384, len(seg))); db = 10 * np.log10(pw + 1e-30)
    f['pwm_5k'] = float(db[(fr > 4990) & (fr < 5010)].max() - np.median(db[((fr > 4900) & (fr < 5100)) & ~((fr > 4985) & (fr < 5015))]))
    band = lambda a, b: float(10 * np.log10(pw[(fr >= a) & (fr < b)].sum() * (fr[1] - fr[0]) + 1e-30))
    lvl = lambda a, b: float(db[(fr >= a) & (fr < b)].mean())
    f['wind_band'] = lvl(3900, 5600) - 0.5 * (lvl(2500, 3500) + lvl(6500, 7500))
    for n, (a, b) in zip(BN, zip(EDGES[:-1], EDGES[1:])): f['band_' + n] = band(a, b)
    f['rms_db'] = float(10 * np.log10(np.mean(seg.astype(np.float64) ** 2) + 1e-30))
    return f


def one_file(args):
    cls, path = args
    a, fs = sf.read(path, dtype='float32'); T = len(a) / fs; c, sh, sh0 = fit_c(a, fs); fn = rpm_model(c, T)
    t = np.arange(len(a)) / fs; r = fn(t); i0, i1 = int(np.searchsorted(r, 5000)), int(np.searchsorted(r, 8000))
    f = features(a[i0:i1], i0 / fs, fs, fn)
    return dict(cls=cls, name=Path(path).name, batch=Path(path).name.split('[')[1].split('_')[1], duration=T, c=c, sharp=sh, sharp_linear=sh0,
                file_rms_db=float(10 * np.log10(np.mean(a.astype(np.float64) ** 2))), **f)


def one_pair(r):
    row = next(csv.DictReader(open(data_path(ROOT, r['info']), encoding='utf-8-sig')))
    name = row['Source_File'].replace('\\', '/').split('/')[-1]; t0, t1 = map(float, row['Source_Interval_Sec'].split('~'))
    src, fs = sf.read(KU / r['sound_class'] / name, dtype='float32'); rec, fs2 = sf.read(data_path(ROOT, r['wav']), dtype='float32')
    c, _, _ = fit_c(src, fs); fn = rpm_model(c, len(src) / fs)
    fo = features(src[round(t0 * fs):round(t1 * fs)], t0, fs, fn); fr_ = features(rec, t0, fs2, fn)
    est = list(map(float, row['Estimated_RPM_Interval'].split('~')))
    return dict(uid=r['uid'], cls=r['sound_class'], pos=r['position'], gain=float(row['Gain']), rpm_fit=[float(fn(t0)), float(fn(t1))], rpm_info=est,
                orig=fo, rec=fr_)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(c, str(p)) for c in CLASSES for p in sorted((KU / c).glob('*.wav'))]
    if (OUT / 'file_features.json').exists(): F = read_json(OUT / 'file_features.json')
    else:
        with ProcessPoolExecutor(12) as ex: F = list(ex.map(one_file, jobs, chunksize=8))
        write_json(OUT / 'file_features.json', F)
    if (OUT / 'paired_1spk.json').exists(): PR = read_json(OUT / 'paired_1spk.json')
    else:
        R1 = read_json(PROJECT / 'runs/ae_paper_single_20261123/manifest.json')
        with ProcessPoolExecutor(12) as ex: PR = list(ex.map(one_pair, R1, chunksize=4))
        write_json(OUT / 'paired_1spk.json', PR)
    from sklearn.metrics import roc_auc_score
    g = collections.defaultdict(list)
    for f in F: g[f['cls']].append(f)
    S = dict(n_files={c: len(g[c]) for c in CLASSES})
    S['rpm_fit'] = {c: dict(c_mean=round(float(np.mean([f['c'] for f in g[c]])), 3), c_std=round(float(np.std([f['c'] for f in g[c]])), 3),
                            sharp=round(float(np.mean([f['sharp'] for f in g[c]])), 1), sharp_linear=round(float(np.mean([f['sharp_linear'] for f in g[c]])), 1),
                            duration=round(float(np.mean([f['duration'] for f in g[c]])), 1)) for c in CLASSES}
    keys = ['file_rms_db', 'rms_db'] + ['band_' + n for n in BN] + ['o4', 'o8', 'o12', 'o16', 'imb_o1', 'imb_o2', 'mod_sb', 'whine_ratio', 'pwm_5k', 'wind_band']
    H = {k: np.array([f[k] for f in g['Healthy']]) for k in keys}
    S['healthy_mean'] = {k: round(float(H[k].mean()), 2) for k in keys}; S['healthy_std'] = {k: round(float(H[k].std()), 2) for k in keys}
    S['class_minus_healthy_db'] = {c: {k: round(float(np.mean([f[k] for f in g[c]]) - H[k].mean()), 2) for k in keys} for c in CLASSES[1:]}
    S['auc_vs_healthy'] = {c: {k: round(float(roc_auc_score(np.r_[np.zeros(len(H[k])), np.ones(len(g[c]))], np.r_[H[k], [f[k] for f in g[c]]])), 3)
                               for k in keys} for c in CLASSES[1:]}
    hb = collections.defaultdict(list)
    for f in g['Healthy']: hb[f['batch']].append(f)
    S['healthy_batches'] = {b: dict(n=len(v), **{k: round(float(np.mean([f[k] for f in v])), 2) for k in ['file_rms_db', 'band_700-1500', 'band_1500-2500', 'o4']})
                            for b, v in sorted(hb.items(), key=lambda kv: int(kv[0]))}
    # paired
    pk = ['band_' + n for n in BN] + ['imb_o1', 'imb_o2', 'mod_sb', 'whine_ratio', 'pwm_5k', 'wind_band', 'o4', 'o8', 'o16']
    S['paired_rpm_check'] = dict(mean_abs_diff_start=round(float(np.mean([abs(p['rpm_fit'][0] - p['rpm_info'][0]) for p in PR])), 1),
                                 mean_abs_diff_end=round(float(np.mean([abs(p['rpm_fit'][1] - p['rpm_info'][1]) for p in PR])), 1))
    S['transfer_rec_minus_orig_db'] = {pos: {n: round(float(np.mean([p['rec']['band_' + n] - p['orig']['band_' + n] for p in PR if p['pos'] == pos])), 1) for n in BN}
                                       for pos in ['FRONT', 'REAR', 'RIGHT']}
    S['signature_orig_vs_rec'] = {}
    for c, k in [('Imbalance', 'imb_o1'), ('Imbalance', 'imb_o2'), ('Modulation', 'mod_sb'), ('Whine', 'whine_ratio'), ('PWM', 'pwm_5k'), ('Wind', 'wind_band')]:
        ph = [p for p in PR if p['cls'] == 'Healthy']; pf = [p for p in PR if p['cls'] == c]
        d = {}
        for side in ('orig', 'rec'):
            h = np.array([p[side][k] for p in ph]); f_ = np.array([p[side][k] for p in pf])
            d[side] = dict(healthy=round(float(h.mean()), 1), fault=round(float(f_.mean()), 1), diff=round(float(f_.mean() - h.mean()), 1),
                           auc=round(float(roc_auc_score(np.r_[np.zeros(len(h)), np.ones(len(f_))], np.r_[h, f_])), 3))
        S['signature_orig_vs_rec'][f'{c}:{k}'] = d
    write_json(OUT / 'summary.json', S); print(json.dumps(S, indent=1, ensure_ascii=False))
