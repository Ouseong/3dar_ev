"""Evaluate the knowledge rules on the source-disjoint test splits (see knowledge_detector.py)."""
from pathlib import Path
import sys, json, collections
import numpy as np
from sklearn.metrics import roc_auc_score
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ev3dar.common import read_json, write_json
OUT = PROJECT / 'runs/knowledge_rules_20261123'
FEATS = ['imb_o1', 'imb_o2', 'mod_sb', 'whine_o16', 'pwm_5k', 'wind_band']
R = read_json(OUT / (sys.argv[1] if len(sys.argv) > 1 else 'roi_features_refined.json'))
X = np.array([[r[f] for f in FEATS] for r in R]); uid = np.array([r['uid'] for r in R]); pos = np.array([r['position'] for r in R])
y = np.array([r['fault'] for r in R]); fc = np.array([r['fault_count'] for r in R]); cls = np.array([r['cls'] for r in R])
# directional contrast: this ROI minus mean of the other two ROIs of the same recording
idx = collections.defaultdict(list)
for i, u in enumerate(uid): idx[u].append(i)
Xd = np.zeros_like(X)
for u, ii in idx.items():
    for i in ii: Xd[i] = X[i] - X[[j for j in ii if j != i]].mean(0)
res = {}
for name, F in [('raw', X), ('directional', Xd), ('raw+directional', np.concatenate([X, Xd], 1))]:
    out = collections.defaultdict(list)
    for seed in (42, 43, 44):
        sp = read_json(PROJECT / f'runs/strict_split_20261123/seed{seed}.json')
        tr, te = np.isin(uid, sp['train']), np.isin(uid, sp['test'])
        Z = np.zeros_like(F)
        for p in np.unique(pos):
            ref = tr & (pos == p) & (fc == 0)                         # clean Healthy train ROIs only
            m, s = F[ref].mean(0), F[ref].std(0) + 1e-6; k = pos == p; Z[k] = (F[k] - m) / s
        score = Z.max(1); thr = 3.0
        out['auc_all'].append(roc_auc_score(y[te], score[te]))
        h = te & ((y == 1) | (fc > 0)); out['auc_hard'].append(roc_auc_score(y[h], score[h]))
        pred = score > thr
        out['recall@z3'].append(pred[te & (y == 1)].mean()); out['specificity@z3'].append(1 - pred[te & (y == 0)].mean())
        out['fp_clean@z3'].append(pred[te & (y == 0) & (fc == 0)].mean()); out['fp_leak@z3'].append(pred[te & (y == 0) & (fc > 0)].mean())
        for c in ['Imbalance', 'Modulation', 'Whine', 'PWM', 'Wind']: out['recall_' + c].append(pred[te & (cls == c)].mean())
        if name == 'raw':
            for j, f in enumerate(FEATS): out['auc_single_' + f].append(roc_auc_score(y[te], Z[te, j]))
    res[name] = {k: dict(seeds=[round(v, 3) for v in vs], mean=round(float(np.mean(vs)), 3)) for k, vs in out.items()}
    print('==', name); [print('  %-24s %s  mean %.3f' % (k, ' / '.join('%.3f' % v for v in vs), np.mean(vs))) for k, vs in out.items()]
write_json(OUT / ('evaluation_' + (sys.argv[1] if len(sys.argv) > 1 else 'roi_features_refined.json')), res)
