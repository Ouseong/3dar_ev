"""Two more maps for Imbalance and Modulation (user request 2026-11-24), added to inputs.npz -> inputs5.npz.
Both faults are order components at low frequency, so they are measured in the ORDER domain: the audio (low-passed at
2 kHz) of each 60-degree mic-angle sector is resampled to uniform shaft angle with the refined RPM and Fourier analysed.
  imb (6 sectors x 101): orders 0.5-2.5 in 0.02 steps (Imbalance adds orders 1 and 2)
  mod (6 sectors x 81) : orders 7.6-8.4 in 0.01 steps (Modulation adds sidebands at 7.9 / 8.1)
dB minus the median of the row; rows repeated x6 to give 36 angle bins like the other maps.
60-degree sectors (~0.66 s, 60-85 revolutions) are needed to resolve 0.1-order sidebands.
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
ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
ADJ = {r['uid']: r['rpm_adj'] for r in json.load(open(PROJECT / 'runs/knowledge_rules_20261123/roi_features_refined.json'))}
GI = np.arange(0.5, 2.5001, 0.02); GM = np.arange(7.6, 8.4001, 0.01)
def one(r):
    a, fs = sf.read(data_path(ROOT, r['wav']), dtype='float32'); dur = len(a) / fs; at, aa = angle_track(data_path(ROOT, r['angles']), dur)
    r0, r1 = np.median(np.array(r['rpm_intervals']), axis=0); q = ADJ[r['uid']]; fn = K.make_rpm(r0 * (1 + q[0]), r1 * (1 + q[1]), q[2], dur)
    b, a_ = signal.butter(6, 2000, fs=fs); x = signal.filtfilt(b, a_, a.astype(np.float64)); I = np.zeros((6, len(GI)), np.float32); M = np.zeros((6, len(GM)), np.float32)
    for s in range(6):
        t0, t1 = np.interp([s * 60, (s + 1) * 60], aa, at); i0, i1 = round(t0 * fs), round(t1 * fs); seg = x[i0:i1]; t = (i0 + np.arange(len(seg))) / fs
        rev = np.concatenate([[0], np.cumsum(fn(t[:-1]) / 60 / fs)]); m = int(rev[-1] * 64); xr = np.interp(np.arange(m) / 64, rev, seg)
        P = 10 * np.log10(np.abs(np.fft.rfft(xr * np.hanning(m))) ** 2 + 1e-30); o = np.fft.rfftfreq(m, 1 / 64)
        vi = np.interp(GI, o, P); vm = np.interp(GM, o, P); I[s] = vi - np.median(vi); M[s] = vm - np.median(vm)
    return np.repeat(I, 6, 0), np.repeat(M, 6, 0)
if __name__ == '__main__':
    R = load_manifest(PROJECT / 'runs/baseline'); out = PROJECT / 'runs/fault_net_20261124'; z = np.load(out / 'inputs.npz')
    with ProcessPoolExecutor(12) as ex: res = list(ex.map(one, R, chunksize=8))
    np.savez(out / 'inputs5.npz', uid=z['uid'], wind=z['wind'], pwm=z['pwm'], whine=z['whine'], imb=np.stack([x[0] for x in res]), mod=np.stack([x[1] for x in res]))
    print('ok', np.stack([x[0] for x in res]).shape, np.stack([x[1] for x in res]).shape)
