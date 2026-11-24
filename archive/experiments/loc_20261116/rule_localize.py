"""Experiment 1: training-free localization from full-rotation angular profiles.

Reference R (freq x angle) = mean of TRAIN all-healthy recordings (removes the common pattern).
D = X - R ; D' = D - mean_over_angle(D)  (angular shape only)
score(pos) = sum_f w_f * mean_{theta in ROI(pos)} D'_f(theta),  w_f = mean_theta D_f within band (excess sign/size)
Band chosen on TRAIN 1-fault recordings only; test reported once.
"""
import json, sys
from pathlib import Path
import numpy as np

PKG = Path(__file__).resolve().parents[2]
BASE = PKG / 'runs/baseline'; OUT = PKG / 'runs/exp_fullrot_20261116'
POS = ['FRONT', 'REAR', 'RIGHT']
ROI = {'FRONT': (135, 225), 'RIGHT': (225, 315), 'REAR': (315, 405)}

z = np.load(OUT / 'fullrot_features.npz'); X = z['X'].astype(np.float64); fc = z['mel_centres_hz']
uids = list(z['uids']); idx = {u: i for i, u in enumerate(uids)}
recs = {r['uid']: r for r in (json.loads(l) for l in open(BASE / 'manifest.jsonl', encoding='utf-8'))}
ang = (np.arange(90) * 4 + 2).astype(float)
roi_mask = {p: ((ang >= a) & (ang < b)) | ((ang + 360 >= a) & (ang + 360 < b)) for p, (a, b) in ROI.items()}
BANDS = [(50, 500), (500, 2000), (2000, 6000), (6000, 12000), (500, 12000), (1000, 8000), (2000, 12000), (50, 20000)]


def scores(us, R, band):
    m = (fc >= band[0]) & (fc < band[1])
    D = X[[idx[u] for u in us]] - R
    Dp = D - D.mean(2, keepdims=True)
    w = D[:, m].mean(2)                                   # (n, f) excess per freq
    out = np.stack([(w[:, :, None] * Dp[:, m][:, :, roi_mask[p]]).sum(1).mean(1) for p in POS], 1)
    return out


def truth(u): return np.array([c != 'Healthy' for c in recs[u]['classes']])


def eval_rows(us, S):
    res = {}
    for k in (1, 2):
        sel = [i for i, u in enumerate(us) if recs[u]['fault_count'] == k]
        hit = 0; per = {p: [0, 0] for p in POS}
        for i in sel:
            t = truth(us[i]); top = np.argsort(-S[i])[:k]
            ok = set(top) == set(np.where(t)[0]); hit += ok
            if k == 1: p = POS[int(np.where(t)[0][0])]; per[p][0] += ok; per[p][1] += 1
        res[k] = (hit, len(sel), per)
    return res


report = {}
for seed in (42, 43, 44):
    sp = json.load(open(BASE / f'splits/seed{seed}.json'))
    healthy_train = [u for u in sp['train'] if recs[u]['fault_count'] == 0]
    R = X[[idx[u] for u in healthy_train]].mean(0)
    # choose band on train 1-fault accuracy
    best = max(BANDS, key=lambda b: eval_rows(sp['train'], scores(sp['train'], R, b))[1][0])
    rep = {}
    for part in ('train', 'validation', 'test'):
        r = eval_rows(sp[part], scores(sp[part], R, best))
        rep[part] = {f'fault{k}': dict(correct=r[k][0], n=r[k][1], acc=r[k][0] / r[k][1]) for k in (1, 2)}
        rep[part]['fault1_by_true_pos'] = {p: f'{v[0]}/{v[1]}' for p, v in r[1][2].items()}
    allband = {f'{b[0]}-{b[1]}': eval_rows(sp['train'], scores(sp['train'], R, b))[1][0] / 240 for b in BANDS}
    report[seed] = dict(chosen_band=best, train_fault1_acc_by_band=allband, **rep)
    print(seed, 'band', best, {p: (rep[p]['fault1']['acc'], rep[p]['fault2']['acc']) for p in rep},
          'test1 by pos', rep['test']['fault1_by_true_pos'])
    print('   train fault1 acc by band', {k: round(v, 3) for k, v in allband.items()})
json.dump(report, open(OUT / 'exp1_rule_localization.json', 'w'), indent=1, default=str)
