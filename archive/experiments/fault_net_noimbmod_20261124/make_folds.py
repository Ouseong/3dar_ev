"""Repeated source-disjoint splits for recordings WITHOUT Imbalance/Modulation (user request 2026-11-24).
A full source-disjoint 5-fold is impossible (healthy files reused up to 12x, across positions; one 394-recording
component). Per repetition: pick a set T of original files (any speaker) by growing it from not-yet-tested
recordings; test = recordings whose 3 files are all in T; train pool = recordings with NO file in T; the rest is
unused in that repetition. Zero source-file overlap between test and train pool is asserted.
Output: runs/fault_net_noimbmod_20261124/folds.json
"""
import sys, json, numpy as np
from pathlib import Path
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_manifest
OUT = PROJECT / 'runs/fault_net_noimbmod_20261124'; OUT.mkdir(parents=True, exist_ok=True)
R = [r for r in load_manifest(PROJECT / 'runs/baseline') if not ({'Imbalance', 'Modulation'} & set(r['classes'][:3]))]
files = sorted({x for r in R for x in r['source_files']}); fi = {f: i for i, f in enumerate(files)}
E = np.array([[fi[x] for x in r['source_files']] for r in R]); n = len(R); fc = np.array([r['fault_count'] for r in R])
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 20; TFILES = int(sys.argv[2]) if len(sys.argv) > 2 else 90
rng = np.random.default_rng(0); tested = np.zeros(n, int); reps = []
for rep in range(NREP):
    inT = np.zeros(len(files), bool)
    w = 1.0 / (1 + tested) ** 2
    for e in rng.choice(n, n, replace=False, p=w / w.sum()):
        if inT.sum() >= TFILES: break
        inT[E[e]] = True
    te = np.where(inT[E].all(1))[0]; tr = np.where(~inT[E].any(1))[0]
    assert not (set(E[te].ravel()) & set(E[tr].ravel()))
    tested[te] += 1; reps.append(dict(test=[R[i]['uid'] for i in te], train_pool=[R[i]['uid'] for i in tr]))
    print(f'rep{rep}: test {len(te)} fc {np.bincount(fc[te], minlength=4)} | train {len(tr)} fc {np.bincount(fc[tr], minlength=4)}')
print('한 번 이상 시험된 녹음', (tested > 0).sum(), '/', n, 'fc', np.bincount(fc[tested > 0], minlength=4))
json.dump(dict(n_kept=n, reps=reps), open(OUT / 'folds.json', 'w'), indent=1)
