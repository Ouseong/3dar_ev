"""Explore Wind-detector variants on the cached angle-resolved spectra (see wind_cache.py). Confound-controlled score =
within-(date, fault count) block AUC: RIGHT speaker Wind vs RIGHT speaker Healthy, fc 1 and 2 only."""
from pathlib import Path
import sys, json, collections, itertools
import numpy as np
from sklearn.metrics import roc_auc_score
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_manifest, read_json
z = np.load(PROJECT / 'runs/wind_detect_20261124/angle_spec.npz'); UID = list(z['uid']); A, G, EDG = z['A'], z['G'], z['edges']; CEN = (EDG[:-1] + EDG[1:]) / 2
ix = {u: i for i, u in enumerate(UID)}; ANG = np.arange(36) * 10 + 5
R3 = load_manifest(PROJECT / 'runs/baseline'); R1 = read_json(PROJECT / 'runs/ae_paper_single_20261123/manifest.json')
blocks = collections.defaultdict(lambda: ([], []))
for r in R3:
    if r['fault_count'] in (1, 2): blocks[(r['event_time'][:10], r['fault_count'])][0 if r['classes'][2] == 'Wind' else 1].append(ix[r['uid']])
w1 = [ix[r['uid']] for r in R1 if r['sound_class'] == 'Wind']; h1 = [ix[r['uid']] for r in R1 if r['sound_class'] == 'Healthy' and r['position'] == 'RIGHT']
def ev(s):
    num = den = 0; per = []
    for b, (p, n) in sorted(blocks.items()):
        a = roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[s[p], s[n]]); per.append(a); num += a * len(p) * len(n); den += len(p) * len(n)
    return num / den, per, roc_auc_score(np.r_[np.ones(len(w1)), np.zeros(len(h1))], np.r_[s[w1], s[h1]])
def spec(mode, lo, hi):
    m = (ANG >= lo) & (ANG < hi)
    return 10 * np.log10(A[:, m].mean(1) + 1e-30) if mode == 'A' else G[:, m].mean(1)
def bump(S, core, f1, f2):
    c = (CEN >= core[0]) & (CEN < core[1]); a = (CEN >= f1[0]) & (CEN < f1[1]); b = (CEN >= f2[0]) & (CEN < f2[1])
    return S[:, c].mean(1) - .5 * (S[:, a].mean(1) + S[:, b].mean(1))
rows = []
for mode in 'AG':
    for (lo, hi) in [(0, 360), (225, 315), (240, 300), (230, 290), (250, 310), (200, 330)]:
        for core in [(4400, 4950), (4300, 5100), (4500, 4900)]:
            for f1, f2 in [((3500, 3900), (5600, 6000)), ((3300, 3800), (5700, 6200)), ((3700, 4000), (5500, 5800))]:
                s = bump(spec(mode, lo, hi), core, f1, f2); a, per, o = ev(s); rows.append((a, mode, lo, hi, core, f1, f2, [round(x, 2) for x in per], round(o, 2)))
rows.sort(reverse=True)
print('top variants (in-block AUC, mean-type, angle window, core, flanks, per block [11/02 fc1, 11/02 fc2, 11/09 fc1], 1-speaker)')
for r in rows[:8]: print('  %.3f' % r[0], *r[1:])
print('  ... worst %.3f' % rows[-1][0])
best = rows[0]
# angle profile of the best bump definition: where does Wind show up?
prof = np.stack([bump(G[:, i], best[4], best[5], best[6]) if best[1] == 'G' else bump(10 * np.log10(A[:, i] + 1e-30), best[4], best[5], best[6]) for i in range(36)], 1)
P = np.concatenate([b[0] for b in blocks.values()]); N = np.concatenate([b[1] for b in blocks.values()])
print('bump (Wind rec - no-Wind rec) by mic angle, 3-speaker fault blocks:'); print('  ' + ' '.join('%+.1f' % v for v in (prof[P].mean(0) - prof[N].mean(0))[::2]))
print('same, 1-speaker RIGHT (Wind - Healthy):'); print('  ' + ' '.join('%+.1f' % v for v in (prof[w1].mean(0) - prof[h1].mean(0))[::2]))
np.save(PROJECT / 'runs/wind_detect_20261124/bump_profile.npy', prof)
json.dump(dict(best=[best[0], best[1], best[2], best[3], list(best[4]), list(best[5]), list(best[6]), best[7], best[8]]), open(PROJECT / 'runs/wind_detect_20261124/explore_best.json', 'w'))
