"""Domain-knowledge rule detector (no learning from fault labels). User request 2026-11-23.

Fault signatures from Kunte et al. Sec. 3.3, measured per ROI (3-speaker set, ROIs as baseline):
  Imbalance  : added orders 1 and 2           -> peak level at order 1, 2 over local background (dB)
  Modulation : sidebands at orders 7.9 / 8.1  -> max(P7.9, P8.1) over background around order 8
  Whine      : order 16 amplified             -> order-16 peak over background
  PWM        : 5000 Hz tone (+ orders +-24)   -> 5000 Hz peak over background (plain FFT)
  Wind       : broadband 3.75-5.75 kHz        -> band level minus neighbouring bands
Order spectra by computed order tracking: the ROI audio is resampled to uniform shaft angle using the
ESTIMATED RPM (median of the 3 speakers' source intervals, linear in time; not measured).
RPM refinement (no labels): per recording, the estimated start/end RPM and a quadratic term are adjusted by
  coordinate search to maximise the sharpness of the HEALTHY orders 4/8/12/16 over the whole recording.
Calibration ONLY on train-split Healthy ROIs from recordings with no fault (fault_count 0), per position:
  z_j = (x_j - mean) / std ; score = max_j z_j. Also a 'directional' variant: z of (x_j in this ROI minus
  mean of x_j in the recording's other two ROIs), to suppress sound leaking from other speakers.
Split: runs/strict_split_20261123 (source-disjoint). Outputs: runs/knowledge_rules_20261123/
"""
from pathlib import Path
import sys, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import soundfile as sf
from scipy import signal

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_config, load_manifest, read_json, data_path, write_json
from ev3dar.features import angle_track, circular_segments

ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
OUT = PROJECT / 'runs/knowledge_rules_20261123'
CFG = load_config(PROJECT / 'runs/baseline/config.resolved.json')
FEATS = ['imb_o1', 'imb_o2', 'mod_sb', 'whine_o16', 'pwm_5k', 'wind_band']
REF = ['o4', 'o8', 'o12', 'o16raw']
SPR = 512                                  # samples per revolution after order tracking


def order_spectrum(seg, t0, fs, rpm_fn):
    """Computed order tracking of one ROI segment -> (orders, power dB)."""
    b, a = signal.butter(6, 10000, fs=fs); x = signal.filtfilt(b, a, seg.astype(np.float64))
    t = t0 + np.arange(len(x)) / fs
    rev = np.concatenate([[0], np.cumsum(rpm_fn(t[:-1]) / 60.0 / fs)])      # shaft revolutions
    n_rev = rev[-1]; m = int(n_rev * SPR)
    xr = np.interp(np.arange(m) / SPR, rev, x)
    w = np.hanning(m); P = np.abs(np.fft.rfft(xr * w)) ** 2
    orders = np.fft.rfftfreq(m, 1.0 / SPR)
    return orders, 10 * np.log10(P + 1e-30), n_rev


def peak_over_bg(o, P, centre, half=0.03, bg=(0.15, 0.6), exclude=()):
    pk = P[(o >= centre - half) & (o <= centre + half)].max()
    d = np.abs(o - centre); mb = (d >= bg[0]) & (d <= bg[1])
    for e in exclude: mb &= np.abs(o - e) > half * 2
    return float(pk - np.median(P[mb]))


def roi_features(audio, fs, at, aa, rpm_fn, pos):
    feats, wts = [], []
    for lo, hi in circular_segments(*CFG['rois'][pos]):
        t0, t1 = np.interp([lo, hi], aa, at); i0, i1 = round(t0 * fs), round(t1 * fs)
        seg = audio[i0:i1]
        o, P, nrev = order_spectrum(seg, t0, fs, rpm_fn)
        f = dict(imb_o1=peak_over_bg(o, P, 1.0), imb_o2=peak_over_bg(o, P, 2.0),
                 mod_sb=max(peak_over_bg(o, P, 7.9, bg=(0.15, 0.6), exclude=(8.0, 8.1)),
                            peak_over_bg(o, P, 8.1, bg=(0.15, 0.6), exclude=(8.0, 7.9))),
                 whine_o16=peak_over_bg(o, P, 16.0), o4=peak_over_bg(o, P, 4.0), o8=peak_over_bg(o, P, 8.0),
                 o12=peak_over_bg(o, P, 12.0), o16raw=float(P[(o > 15.97) & (o < 16.03)].max()))
        fr, pw = signal.welch(seg.astype(np.float64), fs, nperseg=min(16384, len(seg)))
        db = 10 * np.log10(pw + 1e-30)
        pk = db[(fr > 4990) & (fr < 5010)].max(); bgm = ((fr > 4900) & (fr < 5100)) & ~((fr > 4985) & (fr < 5015))
        f['pwm_5k'] = float(pk - np.median(db[bgm]))
        band = lambda a, b: db[(fr >= a) & (fr < b)].mean()
        f['wind_band'] = float(band(3900, 5600) - 0.5 * (band(2500, 3500) + band(6500, 7500)))
        feats.append(f); wts.append(t1 - t0)
    w = np.array(wts) / sum(wts)
    return {k: float(sum(wi * fi[k] for wi, fi in zip(w, feats))) for k in feats[0]}


def make_rpm(r0, r1, c, dur):
    return lambda t: (lambda u: r0 + (r1 - r0) * u + c * (r1 - r0) * u * (1 - u))(np.clip(t / dur, 0, 1))


def sharpness(audio, fs, fn):
    o, P, _ = order_spectrum(audio, 0.0, fs, fn)
    return float(np.mean([peak_over_bg(o, P, k, half=0.02) for k in (4, 8, 12, 16)]))


def refine_rpm(audio, fs, r0, r1, dur):
    p = np.array([0.0, 0.0, 0.0]); best = sharpness(audio, fs, make_rpm(r0, r1, 0, dur))
    for step in (0.01, 0.0025):
        improved = True
        while improved:
            improved = False
            for j in range(3):
                for d in (-step, step):
                    q = p.copy(); q[j] += d * (10 if j == 2 else 1)
                    v = sharpness(audio, fs, make_rpm(r0 * (1 + q[0]), r1 * (1 + q[1]), q[2], dur))
                    if v > best + 1e-3: best, p, improved = v, q, True
    return p, best


def one(r):
    audio, fs = sf.read(data_path(ROOT, r['wav']), dtype='float32')
    dur = len(audio) / fs; at, aa = angle_track(data_path(ROOT, r['angles']), dur)
    r0, r1 = np.median(np.array(r['rpm_intervals']), axis=0)
    base = sharpness(audio, fs, make_rpm(r0, r1, 0, dur))
    p, sharp = refine_rpm(audio, fs, r0, r1, dur)
    rpm_fn = make_rpm(r0 * (1 + p[0]), r1 * (1 + p[1]), p[2], dur)
    return [dict(uid=r['uid'], position=pos, cls=r['classes'][i], fault=int(r['classes'][i] != 'Healthy'),
                 fault_count=r['fault_count'], rpm_adj=[float(v) for v in p], sharp_before=base, sharp_after=sharp,
                 **roi_features(audio, fs, at, aa, rpm_fn, pos))
            for i, pos in enumerate(CFG['positions'])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / 'roi_features_refined.json'
    if cache.exists(): rows = read_json(cache)
    else:
        recs = load_manifest(PROJECT / 'runs/baseline')
        with ProcessPoolExecutor(10) as ex: rows = [x for lst in ex.map(one, recs, chunksize=8) for x in lst]
        write_json(cache, rows)
    print('ROIs', len(rows))


if __name__ == '__main__':
    main()
