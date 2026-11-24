"""Per-position PSD vectors for the ID model (read-only raw data; writes runs/exp_fullrot_20261116/psd_features.npz).

For each position ROI (same angular segments as ev3dar.features.extract_record):
  Welch PSD (Hann, nperseg 4096, 50% overlap) of the raw ROI audio segment(s)
  -> power averaged into 64 log-spaced bands (20 Hz - 20 kHz)
  -> REAR's two segments combined by duration-weighted power average
  -> dB.  Shape per recording: (3 positions, 64).
"""
import sys, json
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG))
from ev3dar.features import angle_track, circular_segments
from ev3dar.common import data_path, read_json

ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
BASE = PKG / 'runs/baseline'; OUT = PKG / 'runs/exp_fullrot_20261116'
CFG = read_json(BASE / 'config.resolved.json')
EDGES = np.geomspace(20, 20000, 65)


def psd_vec(seg, fs):
    f, p = signal.welch(seg, fs=fs, window='hann', nperseg=min(4096, len(seg)), noverlap=None)
    out = np.empty(64)
    for i in range(64):
        m = (f >= EDGES[i]) & (f < EDGES[i + 1])
        out[i] = p[m].mean() if m.any() else np.interp(np.sqrt(EDGES[i] * EDGES[i + 1]), f, p)
    return out


def extract(rec):
    x, fs = sf.read(data_path(ROOT, rec['wav']), dtype='float64')
    dur = len(x) / fs
    at, aa = angle_track(data_path(ROOT, rec['angles']), dur)
    res = np.zeros((3, 64))
    for pi, pos in enumerate(CFG['positions']):
        acc = 0; wsum = 0
        for a, b in circular_segments(*CFG['rois'][pos]):
            t0, t1 = np.interp([a, b], aa, at)
            seg = x[round(t0 * fs):round(t1 * fs)]
            acc = acc + psd_vec(seg, fs) * (t1 - t0); wsum += t1 - t0
        res[pi] = 10 * np.log10(acc / wsum + 1e-20)
    return res.astype(np.float32)


if __name__ == '__main__':
    recs = [json.loads(l) for l in open(BASE / 'manifest.jsonl', encoding='utf-8')]
    P = np.stack([extract(r) for r in recs])
    np.savez_compressed(OUT / 'psd_features.npz', psd=P, uids=np.array([r['uid'] for r in recs]), band_edges_hz=EDGES)
    print('saved', P.shape, float(P.min()), float(P.max()))
