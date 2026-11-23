"""Full-rotation (frequency x angle) features for localization experiments.

Read-only use of raw data and runs/baseline (manifest). Writes only to runs/exp_fullrot_20261116/.
Each recording -> log-mel power (160 bands, 50 Hz-20 kHz) averaged into 90 angle bins (4 deg).
"""
import sys, json
from pathlib import Path
import numpy as np, soundfile as sf
from scipy import signal

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG))
from ev3dar.features import angle_track  # reuse validated angle parsing (read-only)
from ev3dar.common import data_path

ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
BASE = PKG / 'runs/baseline'
OUT = PKG / 'runs/exp_fullrot_20261116'
SR, NFFT, HOP, NMEL, FMIN, FMAX, NANG = 48000, 4096, 480, 160, 50.0, 20000.0, 90


def mel_bank():
    hz2mel = lambda x: 2595 * np.log10(1 + np.asarray(x) / 700)
    mel2hz = lambda x: 700 * (10 ** (np.asarray(x) / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(FMIN), hz2mel(FMAX), NMEL + 2))
    f = np.fft.rfftfreq(NFFT, 1 / SR)
    l = (f[None] - pts[:-2, None]) / (pts[1:-1, None] - pts[:-2, None])
    r = (pts[2:, None] - f[None]) / (pts[2:, None] - pts[1:-1, None])
    w = np.maximum(0, np.minimum(l, r))
    # guarantee narrow low bands get at least nearest FFT bin
    for i in range(NMEL):
        if w[i].sum() == 0: w[i, np.argmin(abs(f - pts[i + 1]))] = 1
    return (w / w.sum(1, keepdims=True)).astype(np.float64), pts[1:-1]


def extract(rec, W):
    x, fs = sf.read(data_path(ROOT, rec['wav']), dtype='float64')
    assert fs == SR and x.ndim == 1
    dur = len(x) / fs
    at, aa = angle_track(data_path(ROOT, rec['angles']), dur)
    _, t, P = signal.spectrogram(x, fs=fs, window='hann', nperseg=NFFT, noverlap=NFFT - HOP,
                                 detrend=False, scaling='spectrum', mode='psd')
    M = W @ P                                    # mel power (NMEL, frames)
    ang = np.interp(t, at, aa)                   # frame centre -> sensor angle
    b = np.clip((ang / (360 / NANG)).astype(int), 0, NANG - 1)
    S = np.zeros((NMEL, NANG)); n = np.zeros(NANG)
    np.add.at(S.T, b, M.T); np.add.at(n, b, 1)
    if (n == 0).any():                           # fill empty bins by circular interpolation
        good = np.where(n > 0)[0]
        for k in np.where(n == 0)[0]:
            S[:, k] = S[:, good[np.argmin(np.minimum(abs(good - k), NANG - abs(good - k)))]]; n[k] = 1
    S /= n
    return (10 * np.log10(S + 1e-14)).astype(np.float32)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(BASE / 'manifest.jsonl', encoding='utf-8')]
    W, centres = mel_bank()
    X = np.stack([extract(r, W) for r in recs])
    np.savez_compressed(OUT / 'fullrot_features.npz', X=X, uids=np.array([r['uid'] for r in recs]),
                        mel_centres_hz=centres, angle_bin_deg=360 / NANG)
    print('saved', X.shape, 'dB range', float(X.min()), float(X.max()))


if __name__ == '__main__':
    main()
