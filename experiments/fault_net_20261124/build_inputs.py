"""Inputs for the fault-band network (user request 2026-11-24: an AI model that uses only the bands/angles where the
faults appear). Per 3-speaker recording, three small maps: 36 mic-angle bins (10 deg) x frequency bins, each made
level-free inside the recording (so recording gain / spectral tilt differences are removed):
  wind  (36 x 58): 3.3-6.2 kHz in 50 Hz bins (5 kHz PWM line removed), mean dB over frames, straight line through the
                   flanks (<4.0, >5.5 kHz) subtracted per angle bin
  pwm   (36 x 68): 4.8-5.2 kHz at 5.9 Hz resolution, mean dB minus the median of the angle bin
  whine (36 x 51): +-150 Hz around the order-16 frequency (16 x RPM / 60, RPM = estimated interval refined on the healthy
                   orders, experiments/knowledge_rules_20261123), mean dB minus the median of the angle bin
STFT n_fft 8192, hop 480. Read-only on raw data. Output: runs/fault_net_20261124/inputs.npz
"""
from pathlib import Path
import sys, json
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf
from scipy import signal
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT / 'experiments/knowledge_rules_20261123'))
from ev3dar.common import load_manifest, data_path
from ev3dar.features import angle_track
import knowledge_detector as K
ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV'); NF = 8192
FR = np.fft.rfftfreq(NF, 1 / 48000); DF = FR[1]
EDG = np.arange(3300, 6201, 50); CEN = (EDG[:-1] + EDG[1:]) / 2; OKW = ~((FR > 4975) & (FR < 5025))
WSEL = [np.where((FR >= a) & (FR < b) & OKW)[0] for a, b in zip(EDG[:-1], EDG[1:])]
FLK = (CEN < 4000) | (CEN > 5500); PM = np.vander(CEN / 1000, 2) @ np.linalg.pinv(np.vander(CEN[FLK] / 1000, 2))
PSEL = np.where((FR >= 4800) & (FR < 5200))[0][:68]; OFF = np.arange(-25, 26)
ADJ = {r['uid']: r['rpm_adj'] for r in json.load(open(PROJECT / 'runs/knowledge_rules_20261123/roi_features_refined.json'))}


def one(r):
    a, fs = sf.read(data_path(ROOT, r['wav']), dtype='float32'); dur = len(a) / fs; at, aa = angle_track(data_path(ROOT, r['angles']), dur)
    _, t, p = signal.spectrogram(a.astype(np.float64), fs=fs, window='hann', nperseg=NF, noverlap=NF - 480, scaling='spectrum', mode='psd')
    db = 10 * np.log10(p + 1e-30); b = np.clip((np.interp(t, at, aa) // 10).astype(int), 0, 35)
    r0, r1 = np.median(np.array(r['rpm_intervals']), axis=0); q = ADJ[r['uid']]; rpm = K.make_rpm(r0 * (1 + q[0]), r1 * (1 + q[1]), q[2], dur)(t)
    c16 = np.rint(16 * rpm / 60 / DF).astype(int)
    wind = np.stack([10 * np.log10(p[s].mean(0) + 1e-30) for s in WSEL]); wl = db[c16[None, :] + OFF[:, None], np.arange(len(t))[None, :]]
    W = np.zeros((36, 58), np.float32); P = np.zeros((36, 68), np.float32); H = np.zeros((36, 51), np.float32)
    for i in range(36):
        m = b == i
        if not m.any(): continue
        w = wind[:, m].mean(1); W[i] = w - PM @ w[FLK]
        x = db[PSEL][:, m].mean(1); P[i] = x - np.median(x)
        h = wl[:, m].mean(1); H[i] = h - np.median(h)
    return r['uid'], W, P, H


if __name__ == '__main__':
    R = load_manifest(PROJECT / 'runs/baseline'); out = PROJECT / 'runs/fault_net_20261124'; out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(12) as ex: res = list(ex.map(one, R, chunksize=8))
    np.savez(out / 'inputs.npz', uid=np.array([x[0] for x in res]), wind=np.stack([x[1] for x in res]), pwm=np.stack([x[2] for x in res]), whine=np.stack([x[3] for x in res]))
    print('ok', len(res))
