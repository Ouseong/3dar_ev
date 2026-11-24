"""Source-disjoint ('strict') 8:1:1 split for the 3-speaker set (user request 2026-11-23).

Problem: recordings share playback source files; the source-sharing graph has a 930-recording component,
so no recording-level split can be source-disjoint while keeping every recording.
Method: assign every SOURCE FILE to train/validation/test; a recording is kept only if all 3 of its sources
fall in the same split, otherwise it is DROPPED (never used anywhere). Simulated annealing maximizes the number
of kept recordings subject to: kept val ~= kept test ~= 10% of kept, and every (fault_count, band) stratum and
every fault class present in val/test. Guarantees zero source-file overlap between splits (asserted).
Output: runs/strict_split_20261123/seed{42,43,44}.json
"""
from pathlib import Path
import sys, json, collections
import numpy as np
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_manifest, write_json

R = load_manifest(PROJECT / 'runs/baseline')
src = sorted({s for r in R for s in r['source_files']}); sid = {s: i for i, s in enumerate(src)}
RS = np.array([[sid[s] for s in r['source_files']] for r in R])            # (N,3)
strata = np.array([r['fault_count'] * 2 + (r['band'][0] == 'B') for r in R])  # 8 strata
by_src = collections.defaultdict(list)
for i, row in enumerate(RS):
    for s in set(row): by_src[s].append(i)
CLS = ['Imbalance', 'Modulation', 'Whine', 'PWM', 'Wind']
has_cls = np.array([[c in r['classes'] for c in CLS] for r in R])


def status(assign):
    a = assign[RS]; same = (a[:, 0] == a[:, 1]) & (a[:, 1] == a[:, 2])
    return np.where(same, a[:, 0], -1)


def score(st):
    kept = (st >= 0).sum(); n = [(st == k).sum() for k in range(3)]
    pen = abs(n[1] - .1 * kept) + abs(n[2] - .1 * kept)
    for k in (1, 2):
        m = st == k
        pen += 20 * sum(((strata[m] == s).sum() < 3) for s in range(8))
        pen += 20 * sum((has_cls[m, j].sum() < 3) for j in range(5))
    return kept - 3 * pen


def anneal(seed, iters=60000):
    rng = np.random.default_rng(seed)
    assign = rng.choice(3, len(src), p=[.8, .1, .1])
    st = status(assign); cur = score(st); best = (cur, assign.copy())
    for t in range(iters):
        T = 5 * (1 - t / iters) + .05
        s = rng.integers(len(src)); old = assign[s]; new = rng.choice([k for k in range(3) if k != old])
        assign[s] = new; st2 = st.copy(); idx = by_src[s]
        a = assign[RS[idx]]; same = (a[:, 0] == a[:, 1]) & (a[:, 1] == a[:, 2]); st2[idx] = np.where(same, a[:, 0], -1)
        sc = score(st2)
        if sc >= cur or rng.random() < np.exp((sc - cur) / T): st, cur = st2, sc
        else: assign[s] = old
        if cur > best[0]: best = (cur, assign.copy())
    return best[1]


if __name__ == '__main__':
    out = PROJECT / 'runs/strict_split_20261123'; out.mkdir(parents=True, exist_ok=True)
    for seed in (42, 43, 44):
        assign = anneal(seed); st = status(assign)
        sp = {k: sorted(R[i]['uid'] for i in np.where(st == j)[0]) for j, k in enumerate(['train', 'validation', 'test'])}
        sp['dropped'] = sorted(R[i]['uid'] for i in np.where(st < 0)[0])
        lk = {r['uid']: r for r in R}
        ss = {k: {s for u in sp[k] for s in lk[u]['source_files']} for k in ['train', 'validation', 'test']}
        ov = {f'{a}__{b}': len(ss[a] & ss[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
        assert all(v == 0 for v in ov.values()), ov
        desc = {k: dict(n=len(sp[k]), fault_count=dict(collections.Counter(lk[u]['fault_count'] for u in sp[k])),
                        classes=dict(collections.Counter(c for u in sp[k] for c in lk[u]['classes'])))
                for k in ['train', 'validation', 'test', 'dropped']}
        write_json(out / f'seed{seed}.json', dict(seed=seed, mode='source_disjoint_strict', source_file_overlap=ov, summary=desc, **sp))
        print(seed, {k: v['n'] for k, v in desc.items()}, 'test fc', desc['test']['fault_count'], 'test cls', desc['test']['classes'])
