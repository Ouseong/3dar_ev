"""Unsupervised within-recording anomaly score (user request 2026-11-23, option C).

Idea: every recording also contains an EMPTY direction (45-135 deg, no speaker). Its sound = room + leakage from all
speakers + recording-time conditions. Subtracting it from each speaker sector cancels recording-level differences.
  d_p = feature(sector p) - feature(empty sector 45-135)
Healthy reference = 0-fault recordings only. Score of a sector = Mahalanobis distance (Ledoit-Wolf) of d_p from the
healthy distribution of the same position. Fault labels are used ONLY for evaluation.
Strictness: the reference for a recording excludes every 0-fault recording sharing any source file with it
(leave-source-out), so no source file of the evaluated recording is in its own reference.
Feature sets:
  mel  : time-averaged 64-band log-mel spectrum (dB) of the sector (baseline feature config)
  know : 6 fault-signature features of experiments/knowledge_rules_20261123 (order 1/2/7.9-8.1/16 with refined RPM,
         5 kHz, 3.9-5.6 kHz band)
Each is scored with ('within') and without ('raw') subtracting the empty sector.
Outputs: runs/within_recording_20261123/{sector_features.json, scores.csv, summary.json}
"""
from pathlib import Path
import sys, json, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import soundfile as sf

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT / 'experiments/knowledge_rules_20261123'))
from ev3dar.common import load_config, load_manifest, data_path, read_json, write_json, write_csv
from ev3dar.features import angle_track, circular_segments, logmel
import knowledge_detector as K

ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
OUT = PROJECT / 'runs/within_recording_20261123'
CFG = load_config(PROJECT / 'runs/baseline/config.resolved.json')
SECT = {'FRONT': (135., 225.), 'REAR': (315., 405.), 'RIGHT': (225., 315.), 'BASE': (45., 135.)}
K.CFG['rois'] = dict(K.CFG['rois'], BASE=[45., 135.])
KF = ['imb_o1', 'imb_o2', 'mod_sb', 'whine_o16', 'pwm_5k', 'wind_band']
RPM_ADJ = {r['uid']: r['rpm_adj'] for r in read_json(PROJECT / 'runs/knowledge_rules_20261123/roi_features_refined.json')}


def one(r):
    audio, fs = sf.read(data_path(ROOT, r['wav']), dtype='float32')
    dur = len(audio) / fs; at, aa = angle_track(data_path(ROOT, r['angles']), dur)
    out = {}
    for name, (lo, hi) in SECT.items():
        P, L = 0, 0
        for a, b in circular_segments(lo, hi):
            t0, t1 = np.interp([a, b], aa, at); seg = audio[round(t0 * fs):round(t1 * fs)]
            db, _ = logmel(seg, CFG); P = P + (10 ** (db / 10)).mean(1) * len(seg); L += len(seg)
        out[name] = dict(mel=(10 * np.log10(P / L + 1e-12)).tolist())
    r0, r1 = np.median(np.array(r['rpm_intervals']), axis=0); q = RPM_ADJ[r['uid']]
    rpm_fn = K.make_rpm(r0 * (1 + q[0]), r1 * (1 + q[1]), q[2], dur)
    for name in SECT:
        f = K.roi_features(audio, fs, at, aa, rpm_fn, name); out[name]['know'] = [f[k] for k in KF]
    return r['uid'], out


def main():
    OUT.mkdir(parents=True, exist_ok=True); recs = load_manifest(PROJECT / 'runs/baseline')
    cache = OUT / 'sector_features.json'
    if cache.exists(): F = read_json(cache)
    else:
        with ProcessPoolExecutor(10) as ex: F = dict(ex.map(one, recs, chunksize=8))
        write_json(cache, F)
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import roc_auc_score
    lk = {r['uid']: r for r in recs}; P3 = ['FRONT', 'REAR', 'RIGHT']
    healthy = [r for r in recs if r['fault_count'] == 0]
    src = {r['uid']: set(r['source_files']) for r in recs}
    rows = []
    variants = [(feat, mode) for feat in ('mel', 'know') for mode in ('within', 'raw')]
    X = {}
    for feat in ('mel', 'know'):
        for mode in ('within', 'raw'):
            X[(feat, mode)] = {u: {p: np.array(F[u][p][feat]) - (np.array(F[u]['BASE'][feat]) if mode == 'within' else 0)
                                   for p in P3} for u in F}
    # reference fits are cached per set of excluded healthy recordings
    for r in recs:
        ref = [h['uid'] for h in healthy if h['uid'] != r['uid'] and not (src[h['uid']] & src[r['uid']])]
        for i, p in enumerate(P3):
            row = dict(uid=r['uid'], position=p, fault_count=r['fault_count'], cls=r['classes'][i],
                       fault=int(r['classes'][i] != 'Healthy'), date=r['event_time'][:10], n_ref=len(ref))
            for v in variants:
                H = np.stack([X[v][u][p] for u in ref]); lw = LedoitWolf().fit(H)
                row[f'{v[0]}_{v[1]}'] = float(np.sqrt(lw.mahalanobis(X[v][r['uid']][p][None])[0]))
            rows.append(row)
    write_csv(OUT / 'scores.csv', rows)
    summ = {}
    for v in variants:
        key = f'{v[0]}_{v[1]}'; s = np.array([x[key] for x in rows]); y = np.array([x['fault'] for x in rows])
        fc = np.array([x['fault_count'] for x in rows]); pos = np.array([x['position'] for x in rows]); uid = np.array([x['uid'] for x in rows])
        res = dict(auc_all=roc_auc_score(y, s))
        m = (fc == 1) | (fc == 2); res['auc_1_2_fault_recordings'] = roc_auc_score(y[m], s[m])
        hard = (y == 1) | (fc > 0); res['auc_healthy_only_from_fault_recordings'] = roc_auc_score(y[hard], s[hard])
        for p in P3:
            k = m & (pos == p); res[f'auc_1_2_within_{p}'] = roc_auc_score(y[k], s[k])
        top, fr = [], []
        for u in np.unique(uid[fc == 1]):
            ii = np.where(uid == u)[0]; top.append(y[ii[np.argmax(s[ii])]])
            f = pos[ii][y[ii] == 1][0]
            if f in ('FRONT', 'REAR'):
                o = 'REAR' if f == 'FRONT' else 'FRONT'
                fr.append(s[ii][pos[ii] == f][0] > s[ii][pos[ii] == o][0])
        res['fc1_fault_sector_ranked_first'] = float(np.mean(top)); res['fc1_front_vs_rear_correct'] = float(np.mean(fr))
        cls = np.array([x['cls'] for x in rows])
        for c in ['Imbalance', 'Modulation', 'Whine', 'PWM', 'Wind']:
            k = (cls == c) | ((y == 0) & (fc == 0)); res['auc_' + c + '_vs_clean_healthy'] = roc_auc_score(y[k], s[k])
        summ[key] = {a: round(float(b), 3) for a, b in res.items()}
    write_json(OUT / 'summary.json', summ)
    for k, v in summ.items(): print('==', k); [print('   %-42s %.3f' % (a, b)) for a, b in v.items()]


if __name__ == '__main__':
    main()
