"""3-speaker recordings: recorded band level vs level PREDICTED from the three original source segments that were played
(power sum of the exact intervals in the info CSV x Gain^2). transfer = recorded - predicted (dB).
If the healthy/fault difference seen in the recordings came from the source files, transfer would be the same for all
recordings; a transfer that changes with the recording block means the recording chain / environment changed.
Output: runs/ku_original_20261123/transfer_3spk.json
"""
from pathlib import Path
import sys, csv, json, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf
from scipy import signal
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_manifest, data_path, write_json
KU = Path('/home/kos/Desktop/3DAR_EV/KU_Leuven_original'); ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
EDGES = [100, 300, 700, 1500, 2500, 3750, 5750, 8000]; BN = [f'{a}-{b}' for a, b in zip(EDGES[:-1], EDGES[1:])]
def bands(x, fs):
    f, p = signal.welch(x.astype(np.float64), fs, nperseg=8192); df = f[1] - f[0]
    return np.array([p[(f >= a) & (f < b)].sum() * df for a, b in zip(EDGES[:-1], EDGES[1:])])
def one(r):
    rec, fs = sf.read(data_path(ROOT, r['wav']), dtype='float32'); pred = 0
    for row in csv.DictReader(open(data_path(ROOT, r['info']), encoding='utf-8-sig')):
        name = row['Source_File'].replace('\\', '/').split('/')[-1]; t0, t1 = map(float, row['Source_Interval_Sec'].split('~'))
        s, fs2 = sf.read(KU / row['Sound_Class'] / name, start=round(t0 * 48000), stop=round(t1 * 48000), dtype='float32')
        pred = pred + bands(s, fs2) * float(row['Gain']) ** 2
    return dict(uid=r['uid'], fc=r['fault_count'], t=r['event_time'][:16], rec=(10 * np.log10(bands(rec, fs))).tolist(), pred=(10 * np.log10(pred)).tolist())
if __name__ == '__main__':
    R = load_manifest(PROJECT / 'runs/baseline')
    with ProcessPoolExecutor(12) as ex: T = list(ex.map(one, R, chunksize=8))
    write_json(PROJECT / 'runs/ku_original_20261123/transfer_3spk.json', T)
    g = collections.defaultdict(list)
    for x in T: g[(x['t'][:10], x['fc'])].append(x)
    print('%-22s %4s  ' % ('block', 'n') + ' '.join('%9s' % b for b in BN))
    for k in sorted(g):
        v = g[k]; rec = np.mean([x['rec'] for x in v], 0); pr = np.mean([x['pred'] for x in v], 0)
        print('%-22s %4d  REC  ' % (f'{k[0]} fc{k[1]} {min(x["t"][11:] for x in v)}-{max(x["t"][11:] for x in v)}', len(v)) + ' '.join('%9.1f' % a for a in rec))
        print('%-22s %4s  PRED ' % ('', '') + ' '.join('%9.1f' % a for a in pr))
        print('%-22s %4s  R-P  ' % ('', '') + ' '.join('%9.1f' % a for a in rec - pr))
