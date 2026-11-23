"""Source-disjoint split where TRAIN/VALIDATION contain only all-Healthy (fault_count 0) and all-Fault (fault_count 3)
recordings, and TEST is mainly 1- and 2-fault recordings (plus held-out 0/3 for reference). User request 2026-11-23.
Every source file is assigned to train / validation / test; a recording is kept only if all 3 sources share a split AND
it is eligible for that split (train/val: fault_count 0 or 3; test: any). Simulated annealing maximises
4 x min(train 0-fault, train 3-fault) + kept train + kept test(1/2-fault), with validation ~10% of train and some 0/3 recordings in test. Overlap asserted 0.
Output: runs/extremes_split_20261123/seed{42,43,44}.json
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
ext = (fc == 0) | (fc == 3)
by_src = collections.defaultdict(list)
for i, row in enumerate(RS):
    for s in set(row): by_src[s].append(i)


def status(assign, idx=None):
    rs = RS if idx is None else RS[idx]; e = ext if idx is None else ext[idx]
    a = assign[rs]; same = (a[:, 0] == a[:, 1]) & (a[:, 1] == a[:, 2]); st = np.where(same, a[:, 0], -1)
    return np.where((st == 2) | ((st >= 0) & e), st, -1)


def score(st):
    ntr, nva = (st == 0).sum(), (st == 1).sum()
    t12 = ((st == 2) & ~ext).sum(); t03 = [((st == 2) & (fc == k)).sum() for k in (0, 3)]
    tr0, tr3 = ((st == 0) & (fc == 0)).sum(), ((st == 0) & (fc == 3)).sum()
    return 4 * min(tr0, tr3) + ntr + t12 - 3 * abs(nva - .1 * (ntr + nva)) - 5 * sum(max(0, 15 - x) for x in t03) \
        - 5 * sum(max(0, 15 - ((st == k) & (fc == f)).sum()) for k in (0, 1) for f in (0, 3))


def anneal(seed, iters=150000):
    rng = np.random.default_rng(seed); assign = rng.choice(3, len(src), p=[.5, .1, .4])
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
    out = PROJECT / 'runs/extremes_split_20261123'; out.mkdir(parents=True, exist_ok=True)
    for seed in (42, 43, 44):
        st = status(anneal(seed))
        sp = {k: sorted(R[i]['uid'] for i in np.where(st == j)[0]) for j, k in enumerate(['train', 'validation', 'test'])}
        lk = {r['uid']: r for r in R}
        ss = {k: {s for u in sp[k] for s in lk[u]['source_files']} for k in sp}
        ov = {f'{a}__{b}': len(ss[a] & ss[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
        assert all(v == 0 for v in ov.values()), ov
        assert all(lk[u]['fault_count'] in (0, 3) for u in sp['train'] + sp['validation'])
        desc = {k: dict(collections.Counter(lk[u]['fault_count'] for u in sp[k])) for k in sp}
        write_json(out / f'seed{seed}.json', dict(seed=seed, mode='source_disjoint_train_on_0_and_3_faults', source_file_overlap=ov,
                                                 summary=desc, dropped=int((st < 0).sum()), **sp))
        print(seed, {k: dict(sorted(v.items())) for k, v in desc.items()}, 'dropped', int((st < 0).sum()))
