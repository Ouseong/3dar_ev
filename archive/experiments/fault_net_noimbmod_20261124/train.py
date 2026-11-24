"""FaultBandNet (3 types) on recordings WITHOUT Imbalance/Modulation, repeated source-disjoint splits
(make_folds.py). Same model / loss / optimiser as experiments/fault_net_20261124/train_fault_net.py.
Validation (early stopping) = 15 % of the train pool grouped by right-speaker file; test never touched.
Per recording, probabilities are averaged over the repetitions in which it was tested.
Output: runs/fault_net_noimbmod_20261124/{probs.npy, tested_uids.json, summary.json}
"""
import sys, json, numpy as np, torch
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT / 'experiments/fault_net_20261124'))
import train_fault_net as T
OUT = PROJECT / 'runs/fault_net_noimbmod_20261124'; F = json.load(open(OUT / 'folds.json'))
ix = {r['uid']: i for i, r in enumerate(T.R)}; dev = torch.device('cuda')
S = np.zeros((T.n, 3, 3)); C = np.zeros(T.n)
for k, rep in enumerate(F['reps']):
    tp = np.array([ix[u] for u in rep['train_pool']]); te = np.array([ix[u] for u in rep['test']])
    a, b = next(GroupShuffleSplit(1, test_size=.15, random_state=k).split(tp, groups=T.grp[tp]))
    S[te] += T.fit_predict(tp[a], tp[b], te, 1000 + k, dev); C[te] += 1
idx = np.where(C > 0)[0]; P = (S[idx] / C[idx, None, None]).astype(np.float32)
np.save(OUT / 'probs.npy', P); json.dump([T.R[i]['uid'] for i in idx], open(OUT / 'tested_uids.json', 'w'))
summ = T.report(P, idx)
# sector-level accuracy by fault count
pred = P.max(2) >= .5; y = T.Y[idx].max(2) > 0; f = T.fc[idx]; sec = {}
for q in range(4):
    m = f == q; H = ~y[m]; Fm = y[m]; p = pred[m]
    sec[f'fc{q}'] = dict(n_rec=int(m.sum()), healthy_sectors=int(H.sum()), healthy_correct=int((~p[H]).sum()),
                         fault_sectors=int(Fm.sum()), fault_correct=int(p[Fm].sum()), acc=float((p == Fm).mean()) if m.any() else None)
summ['sector_by_fc'] = sec; summ['n_tested'] = int(len(idx)); summ['times_tested_mean'] = float(C[idx].mean())
json.dump(summ, open(OUT / 'summary.json', 'w'), indent=1); print(json.dumps(summ, indent=1))
