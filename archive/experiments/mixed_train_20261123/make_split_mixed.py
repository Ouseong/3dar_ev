"""Source-disjoint split where TRAIN/VALIDATION contain only 1-fault and 2-fault recordings; TEST holds unseen 1-/2-fault
recordings (main evaluation) plus some 0-/3-fault recordings for reference. User request 2026-11-23.
Same mechanism as experiments/extremes_train_20261123/make_split_extremes.py (source files assigned to splits,
recordings mixing splits dropped, simulated annealing), objective: kept train + 2 x min(train 1-fault, train 2-fault) with
validation ~10% of train and >= 15 test recordings of each fault count. Overlap asserted 0.
Output: runs/mixed_split_20261123/seed{42,43,44}.json
"""
from pathlib import Path
import sys, collections
import numpy as np
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_manifest, write_json

R = load_manifest(PROJECT / 'runs/baseline')
src = sorted({s for r in R for s in r['source_files']}); sid = {s: i for i, s in enumerate(src)}
RS = np.array([[sid[s] for s in r['source_files']] for r in R]); fc = np.array([r['fault_count'] for r in R])
elig = (fc == 1) | (fc == 2)
by_src = collections.defaultdict(list)
for i, row in enumerate(RS):
    for s in set(row): by_src[s].append(i)


def status(assign, idx=None):
    rs = RS if idx is None else RS[idx]; e = elig if idx is None else elig[idx]
    a = assign[rs]; same = (a[:, 0] == a[:, 1]) & (a[:, 1] == a[:, 2]); st = np.where(same, a[:, 0], -1)
    return np.where((st == 2) | ((st >= 0) & e), st, -1)


def score(st):
    ntr, nva = (st == 0).sum(), (st == 1).sum()
    t = [((st == 2) & (fc == k)).sum() for k in range(4)]
    tr = [((st == 0) & (fc == k)).sum() for k in (1, 2)]
    return ntr + 2 * min(tr) - 3 * abs(nva - .1 * (ntr + nva)) - 10 * max(0, 25 - nva) \
        - 5 * sum(max(0, 15 - x) for x in t) - 5 * sum(max(0, 50 - x) for x in t[1:3])


def anneal(seed, iters=150000):
    rng = np.random.default_rng(seed); assign = rng.choice(3, len(src), p=[.6, .1, .3])
    st = status(assign); cur = score(st); best = (cur, assign.copy())
    for t in range(iters):
        T = 5 * (1 - t / iters) + .05; s = rng.integers(len(src)); old = assign[s]
        assign[s] = rng.choice([k for k in range(3) if k != old]); st2 = st.copy(); idx = by_src[s]; st2[idx] = status(assign, idx)
        sc = score(st2)
        if sc >= cur or rng.random() < np.exp((sc - cur) / T): st, cur = st2, sc
        else: assign[s] = old
        if cur > best[0]: best = (cur, assign.copy())
    return best[1]


if __name__ == '__main__':
    out = PROJECT / 'runs/mixed_split_20261123'; out.mkdir(parents=True, exist_ok=True)
    for seed in (42, 43, 44):
        st = status(anneal(seed))
        sp = {k: sorted(R[i]['uid'] for i in np.where(st == j)[0]) for j, k in enumerate(['train', 'validation', 'test'])}
        lk = {r['uid']: r for r in R}
        ss = {k: {s for u in sp[k] for s in lk[u]['source_files']} for k in sp}
        ov = {f'{a}__{b}': len(ss[a] & ss[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
        assert all(v == 0 for v in ov.values()), ov
        assert all(lk[u]['fault_count'] in (1, 2) for u in sp['train'] + sp['validation'])
        desc = {k: dict(sorted(collections.Counter(lk[u]['fault_count'] for u in sp[k]).items())) for k in sp}
        write_json(out / f'seed{seed}.json', dict(seed=seed, mode='source_disjoint_train_on_1_and_2_faults', source_file_overlap=ov,
                                                 summary=desc, dropped=int((st < 0).sum()), **sp))
        print(seed, desc, 'dropped', int((st < 0).sum()))
