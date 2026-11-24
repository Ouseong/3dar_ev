"""Source-disjoint 8:1:1 split of recordings whose speakers are only Healthy / PWM / Wind (user request 2026-11-23:
"train with Wind and PWM only, mix 0/1/2/3-fault recordings, cut into sectors"). 522 eligible recordings.
Same mechanism as experiments/strict_split_20261123/make_strict_split.py (source files assigned to splits,
recordings mixing splits dropped, simulated annealing): maximise kept recordings, val ~= test ~= 10 %, with minimum fault recordings
(test: 1-fault 12, 2-fault 6, 3-fault 3; validation: 8 / 4 / 2) enforced by a large penalty. Overlap asserted 0.
Output: runs/pwm_wind_split_20261123/seed{42,43,44}.json
"""
from pathlib import Path
import sys, collections
import numpy as np
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_manifest, write_json
R = [r for r in load_manifest(PROJECT / 'runs/baseline') if set(r['classes']) <= {'Healthy', 'PWM', 'Wind'}]
src = sorted({s for r in R for s in r['source_files']}); sid = {s: i for i, s in enumerate(src)}
RS = np.array([[sid[s] for s in r['source_files']] for r in R]); fc = np.array([r['fault_count'] for r in R])
by_src = collections.defaultdict(list)
for i, row in enumerate(RS):
    for s in set(row): by_src[s].append(i)
def status(assign, idx=None):
    a = assign[RS if idx is None else RS[idx]]; same = (a[:, 0] == a[:, 1]) & (a[:, 1] == a[:, 2]); return np.where(same, a[:, 0], -1)
def score(st):
    kept = (st >= 0).sum(); n = [(st == k).sum() for k in range(3)]
    pen = abs(n[1] - .1 * kept) + abs(n[2] - .1 * kept)
    MIN = {2: {1: 12, 2: 6, 3: 3}, 1: {1: 8, 2: 4, 3: 2}}   # minimum fault recordings in test (2) / validation (1)
    for k, m in MIN.items(): pen += 10 * sum(max(0, v - ((st == k) & (fc == f)).sum()) for f, v in m.items())
    return kept - 3 * pen
def anneal(seed, iters=100000):
    rng = np.random.default_rng(seed); assign = rng.choice(3, len(src), p=[.8, .1, .1]); st = status(assign); cur = score(st); best = (cur, assign.copy())
    for t in range(iters):
        T = 5 * (1 - t / iters) + .05; s = rng.integers(len(src)); old = assign[s]
        assign[s] = rng.choice([k for k in range(3) if k != old]); st2 = st.copy(); idx = by_src[s]; st2[idx] = status(assign, idx); sc = score(st2)
        if sc >= cur or rng.random() < np.exp((sc - cur) / T): st, cur = st2, sc
        else: assign[s] = old
        if cur > best[0]: best = (cur, assign.copy())
    return best[1]
if __name__ == '__main__':
    out = PROJECT / 'runs/pwm_wind_split_20261123'; out.mkdir(parents=True, exist_ok=True); lk = {r['uid']: r for r in R}
    for seed in (42, 43, 44):
        st = status(anneal(seed)); sp = {k: sorted(R[i]['uid'] for i in np.where(st == j)[0]) for j, k in enumerate(['train', 'validation', 'test'])}
        ss = {k: {s for u in sp[k] for s in lk[u]['source_files']} for k in sp}
        ov = {f'{a}__{b}': len(ss[a] & ss[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
        assert all(v == 0 for v in ov.values()), ov
        desc = {k: dict(sorted(collections.Counter(lk[u]['fault_count'] for u in v).items())) for k, v in sp.items()}
        write_json(out / f'seed{seed}.json', dict(seed=seed, mode='source_disjoint_pwm_wind_only', source_file_overlap=ov, summary=desc, dropped=int((st < 0).sum()), **sp))
        print(seed, desc, 'dropped', int((st < 0).sum()))
