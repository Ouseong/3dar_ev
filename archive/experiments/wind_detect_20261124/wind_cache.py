"""Cache for Wind-detector development: per recording, 3.3-6.2 kHz spectrum in 50 Hz bins (PWM 5 kHz line removed) for
each 10-degree mic-angle bin. A = mean power over frames (arithmetic), G = mean of dB over frames (robust to transients).
Output: runs/wind_detect_20261124/angle_spec.npz (uids, A[n,36,58], G[n,36,58])
"""
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf
from scipy import signal
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_manifest, read_json, data_path
from ev3dar.features import angle_track
ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV'); NF = 4096
FR = np.fft.rfftfreq(NF, 1 / 48000); EDG = np.arange(3300, 6201, 50); OK = ~((FR > 4975) & (FR < 5025))
SEL = [np.where((FR >= a) & (FR < b) & OK)[0] for a, b in zip(EDG[:-1], EDG[1:])]
def one(r):
    a, fs = sf.read(data_path(ROOT, r['wav']), dtype='float32'); at, aa = angle_track(data_path(ROOT, r['angles']), len(a) / fs)
    _, t, p = signal.spectrogram(a.astype(np.float64), fs=fs, window='hann', nperseg=NF, noverlap=NF - 480, scaling='spectrum', mode='psd')
    B = np.stack([p[s].mean(0) for s in SEL]); b = np.clip((np.interp(t, at, aa) // 10).astype(int), 0, 35)
    A = np.stack([B[:, b == i].mean(1) for i in range(36)]); G = np.stack([(10 * np.log10(B[:, b == i] + 1e-30)).mean(1) for i in range(36)])
    return r['uid'], A, G
if __name__ == '__main__':
    R = load_manifest(PROJECT / 'runs/baseline') + read_json(PROJECT / 'runs/ae_paper_single_20261123/manifest.json')
    with ProcessPoolExecutor(12) as ex: res = list(ex.map(one, R, chunksize=8))
    np.savez(PROJECT / 'runs/wind_detect_20261124/angle_spec.npz', uid=np.array([u for u, _, _ in res]), A=np.stack([a for _, a, _ in res]), G=np.stack([g for _, _, g in res]), edges=EDG)
    print('ok', len(res))
