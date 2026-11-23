"""Is the loss of fault signatures caused by 3DAR cancellation? 1-speaker recordings, paired with the played source segment.
For each recording: time span where the mic faces the speaker (within +-45 deg of FRONT 180 / REAR 0 / RIGHT 270) and the
span where it faces away (opposite +-45 deg). Same signature features (analyze_originals.features) on the recording span
and on the SAME time span of the original. If 3DAR cancellation were the cause, 'facing' would keep the original contrast.
Output: runs/ku_original_20261123/facing_vs_away.json
"""
from pathlib import Path
import sys, csv, json, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(Path(__file__).parent))
from ev3dar.common import read_json, write_json, data_path
from ev3dar.features import angle_track, circular_segments
import analyze_originals as A
CENTRE = {'FRONT': 180, 'REAR': 0, 'RIGHT': 270}
def spans(c): return [((c - 45) % 360, (c - 45) % 360 + 90)]
def one(r):
    row = next(csv.DictReader(open(data_path(A.ROOT, r['info']), encoding='utf-8-sig')))
    name = row['Source_File'].replace('\\', '/').split('/')[-1]; t0, _ = map(float, row['Source_Interval_Sec'].split('~'))
    src, fs = sf.read(A.KU / r['sound_class'] / name, dtype='float32'); rec, _ = sf.read(data_path(A.ROOT, r['wav']), dtype='float32')
    c, _, _ = A.fit_c(src, fs); fn = A.rpm_model(c, len(src) / fs); at, aa = angle_track(data_path(A.ROOT, r['angles']), len(rec) / fs)
    out = dict(cls=r['sound_class'], pos=r['position'])
    for tag, cen in [('facing', CENTRE[r['position']]), ('away', (CENTRE[r['position']] + 180) % 360)]:
        segs = circular_segments((cen - 45) % 360, (cen - 45) % 360 + 90)
        (lo, hi) = max(segs, key=lambda s: s[1] - s[0])                      # longest contiguous part (REAR is split at 0 deg)
        a, b = np.interp([lo, hi], aa, at); i0, i1 = round(a * fs), round(b * fs)
        out[tag] = dict(rec=A.features(rec[i0:i1], t0 + a, fs, fn), orig=A.features(src[round((t0 + a) * fs):round((t0 + b) * fs)], t0 + a, fs, fn), dur=float(b - a))
    return out
if __name__ == '__main__':
    R1 = read_json(PROJECT / 'runs/ae_paper_single_20261123/manifest.json')
    with ProcessPoolExecutor(12) as ex: P = list(ex.map(one, R1, chunksize=4))
    write_json(PROJECT / 'runs/ku_original_20261123/facing_vs_away.json', P)
    from sklearn.metrics import roc_auc_score
    print('%-22s %-7s | %-28s | %-28s' % ('signature', 'span', 'ORIGINAL fault-healthy (AUC)', 'RECORDED fault-healthy (AUC)'))
    for cls, k in [('PWM', 'pwm_5k'), ('Whine', 'whine_ratio'), ('Wind', 'wind_band'), ('Imbalance', 'imb_o2'), ('Imbalance', 'imb_o1')]:
        pos_ok = {p['pos'] for p in P if p['cls'] == cls}
        for tag in ('facing', 'away'):
            s = []
            for side in ('orig', 'rec'):
                h = np.array([p[tag][side][k] for p in P if p['cls'] == 'Healthy' and p['pos'] in pos_ok]); f = np.array([p[tag][side][k] for p in P if p['cls'] == cls])
                s.append('%+6.1f dB (%.2f)' % (f.mean() - h.mean(), roc_auc_score(np.r_[np.zeros(len(h)), np.ones(len(f))], np.r_[h, f])))
            print('%-22s %-7s | %-28s | %-28s' % (f'{cls}:{k}', tag, s[0], s[1]))
