"""Compare variants on identical splits/rules. Fault = 1-P(Healthy) >= 0.5, type = argmax."""
import json, sys, csv, collections
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

PKG = Path(__file__).resolve().parents[2]
BASE = PKG / 'runs/baseline'; OUT = PKG / 'runs/exp_fullrot_20261116'
C = ['Healthy', 'Imbalance', 'Modulation', 'Whine', 'Wind', 'PWM']
recs = {r['uid']: r for r in (json.loads(l) for l in open(BASE / 'manifest.jsonl', encoding='utf-8'))}


def load_variant(v, seed, part='test'):
    if v == 'baseline(original)':
        rows = [r for r in csv.DictReader(open(BASE / f'seed{seed}/test_predictions.csv', encoding='utf-8-sig'))]
        return [dict(uid=r['recording_id'], position=r['position'], gt=r['gt_type'],
                     P=np.array([float(r['prob_' + c]) for c in C])) for r in rows] if part == 'test' else None
    rows = json.load(open(OUT / v / f'seed{seed}/predictions.json'))
    return [dict(uid=r['uid'], position=r['position'], gt=r['gt'], P=np.array([r['p_' + c] for c in C]))
            for r in rows if r['split'] == part]


def metrics(rows):
    y = np.array([C.index(r['gt']) for r in rows]); P = np.stack([r['P'] for r in rows]); p = P.argmax(1)
    gf = (y != 0).astype(int); sc = 1 - P[:, 0]; pf = (sc >= .5).astype(int)
    out = dict(acc=np.mean(p == y), macroF1=f1_score(y, p, labels=range(6), average='macro', zero_division=0),
               faultF1=f1_score(gf, pf, zero_division=0), AUC=roc_auc_score(gf, sc))
    for r, a, b in zip(rows, pf, gf): r['pf'], r['gf'], r['pred'] = a, b, C[int(r['P'].argmax())]
    by = collections.defaultdict(list)
    for r in rows: by[r['uid']].append(r)
    for k in range(4):
        U = [v for u, v in by.items() if recs[u]['fault_count'] == k]
        h = [r for v in U for r in v if not r['gf']]
        out[f'FA{k}'] = np.mean([r['pf'] for r in h]) if h else np.nan
    U1 = [v for u, v in by.items() if recs[u]['fault_count'] == 1]
    out['exact1'] = np.mean([all(r['pf'] == r['gf'] for r in v) for v in U1])
    out['other1'] = np.mean([any(r['pf'] and not r['gf'] for r in v) for v in U1])
    out['miss'] = np.mean([1 - r['pf'] for r in rows if r['gf']])
    for c in ['Imbalance', 'Modulation', 'Whine', 'PWM', 'Wind']:
        R = [r for r in rows if r['gt'] == c]; out['rec_' + c] = np.mean([r['pred'] == c for r in R])
    return out


if __name__ == '__main__':
    variants = ['baseline(original)'] + sys.argv[1:]
    keys = ['acc', 'macroF1', 'faultF1', 'AUC', 'miss', 'FA0', 'FA1', 'FA2', 'exact1', 'other1',
            'rec_Imbalance', 'rec_Modulation', 'rec_Whine', 'rec_PWM', 'rec_Wind']
    print('variant | seed | ' + ' | '.join(keys))
    summary = {}
    for v in variants:
        ms = []
        for s in (42, 43, 44):
            try: m = metrics(load_variant(v, s))
            except FileNotFoundError: continue
            ms.append(m); print(f'{v} | {s} | ' + ' | '.join(f'{m[k]:.3f}' for k in keys))
        if ms:
            summary[v] = {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms], ddof=1)) if len(ms) > 1 else 0) for k in keys}
            print(f'{v} | mean | ' + ' | '.join(f'{summary[v][k][0]:.3f}' for k in keys))
    json.dump(summary, open(OUT / 'compare_summary.json', 'w'), indent=1)
