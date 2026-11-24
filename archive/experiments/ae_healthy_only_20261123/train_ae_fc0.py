"""Normal-only ConvAE trained on ALL-HEALTHY (0-fault) recordings only; does it separate all-healthy vs all-fault?
User request 2026-11-23. Model / features / scoring = experiments/ae_20261119/train_ae.py (ConvAE latent 64, mel 64 x 100
per ROI view, per-mel z-score from train, masked MSE, ROI score = duration-weighted view error).
Split (per seed): 0-fault recordings grouped by shared source files (connected components) and assigned 8:1:1 to
train / validation / test. TEST also gets every 3-fault recording (they contain no Healthy source file, so no source
overlap with train) and every 1-/2-fault recording whose Healthy sources are not in train/validation.
No fault label is used for training or early stopping (validation = Healthy reconstruction only).
Output: runs/ae_healthy_only_20261123/seed*/ {metrics.json, test_scores.csv}
"""
from pathlib import Path
import sys, json, time, collections
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT / 'experiments/ae_20261119'))
import train_ae as base
from ev3dar.common import load_config, load_manifest, write_json, write_csv
from ev3dar.engine import seed_all, GpuThrottle

OUT = PROJECT / 'runs/ae_healthy_only_20261123'


def make_split(records, seed):
    h = [r for r in records if r['fault_count'] == 0]
    par = list(range(len(h)))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    first = {}
    for i, r in enumerate(h):
        for s in r['source_files']:
            if s in first: par[f(i)] = f(first[s])
            else: first[s] = i
    comps = collections.defaultdict(list)
    for i in range(len(h)): comps[f(i)].append(h[i]['uid'])
    groups = list(comps.values()); rng = np.random.default_rng(seed); rng.shuffle(groups)
    groups.sort(key=len, reverse=True)                       # place big components first (greedy fill)
    sp = dict(train=[], validation=[], test=[]); target = dict(train=.8, validation=.1, test=.1)
    for g in groups:
        k = min(sp, key=lambda k: len(sp[k]) / (target[k] * len(h)))
        sp[k] += g
    lk = {r['uid']: r for r in records}
    used = {s for k in ('train', 'validation') for u in sp[k] for s in lk[u]['source_files']}
    for r in records:
        if r['fault_count'] == 3 or (r['fault_count'] in (1, 2) and not (set(r['source_files']) & used)):
            sp['test'].append(r['uid'])
    ss = {k: {s for u in sp[k] for s in lk[u]['source_files']} for k in sp}
    ov = {f'{a}__{b}': len(ss[a] & ss[b]) for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}
    assert all(v == 0 for v in ov.values()), ov
    return sp, ov


def run(seed, cfg, records, device):
    seed_all(seed); t0 = time.time(); sp, ov = make_split(records, seed)
    lk = {r['uid']: r for r in records}
    get = lambda k: base.load_rois([lk[u] for u in sp[k]], cfg)
    Vtr, Mtr, Dtr, _ = get('train'); Vva, Mva, Dva, _ = get('validation'); Vte, Mte, Dte, meta = get('test')
    mean, std = base.fit_stats(Vtr, Mtr)
    Xtr, Ktr, _, _ = base.to_views(Vtr, Mtr, Dtr, mean, std); Xva, Kva, _, _ = base.to_views(Vva, Mva, Dva, mean, std)
    Xte, Kte, Wte, Ite = base.to_views(Vte, Mte, Dte, mean, std)
    model = base.ConvAE(64).to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    g = torch.Generator().manual_seed(seed); thr = GpuThrottle(); best, stale, hist = 1e9, 0, []
    sd = OUT / f'seed{seed}'; sd.mkdir(parents=True, exist_ok=True)
    for ep in range(1, 101):
        model.train(); perm = torch.randperm(len(Xtr), generator=g); thr.reset()
        for i in range(0, len(perm), 64):
            idx = perm[i:i + 64]; x = Xtr[idx].to(device); m = Ktr[idx].to(device); rec, _ = model(x)
            loss = ((rec - x) ** 2 * m).sum() / (m.sum() * x.shape[2]); opt.zero_grad(); loss.backward(); opt.step(); thr.step()
        v = float(base.view_errors(model, Xva, Kva, device).mean()); hist.append(dict(epoch=ep, val_healthy_recon=v))
        if v < best: best, stale = v, 0; torch.save(model.state_dict(), sd / 'ae_best.pt')
        else:
            stale += 1
            if stale >= 15: break
    model.load_state_dict(torch.load(sd / 'ae_best.pt'))
    s = base.roi_scores(base.view_errors(model, Xte, Kte, device), Wte, Ite, len(meta))
    fc = np.array([m['fault_count'] for m in meta]); y = np.array([m['fault'] for m in meta]); pos = np.array([m['position'] for m in meta])
    res = dict(seed=seed, n_recordings={k: len(v) for k, v in sp.items()}, source_overlap=ov, best_epoch=int(np.argmin([h['val_healthy_recon'] for h in hist]) + 1),
               test_rois_by_fault_count={str(k): int((fc == k).sum()) for k in range(4)})
    res['auc_all_healthy_vs_all_fault_recordings'] = float(roc_auc_score(fc[(fc == 0) | (fc == 3)] == 3, s[(fc == 0) | (fc == 3)]))
    k = (fc == 0) | ((fc > 0) & (y == 0)); res['auc_clean_healthy_vs_healthy_quarter_in_fault_recordings'] = float(roc_auc_score(fc[k] > 0, s[k]))
    m12 = (fc == 1) | (fc == 2)
    res['auc_within_1_2_fault_recordings'] = float(roc_auc_score(y[m12], s[m12])) if m12.sum() and 0 < y[m12].mean() < 1 else None
    res['mean_score_by_group'] = {'clean_healthy': float(s[fc == 0].mean()), 'healthy_q_in_fault_rec': float(s[(fc > 0) & (y == 0)].mean()),
                                  'fault_q': float(s[y == 1].mean())}
    thr_v = float(np.percentile(base.roi_scores(base.view_errors(model, Xva, Kva, device), *base.to_views(Vva, Mva, Dva, mean, std)[2:], len(Vva)), 95))
    res['threshold_p95_val_healthy'] = thr_v
    res['flagged_rate'] = {'clean_healthy': float((s[fc == 0] >= thr_v).mean()), '3fault': float((s[fc == 3] >= thr_v).mean()),
                           'healthy_q_in_1_2': float((s[m12 & (y == 0)] >= thr_v).mean()), 'fault_q_in_1_2': float((s[m12 & (y == 1)] >= thr_v).mean())}
    res['seconds'] = time.time() - t0
    write_json(sd / 'metrics.json', res)
    write_csv(sd / 'test_scores.csv', [dict(**mm, score=float(v)) for mm, v in zip(meta, s)])
    print(json.dumps(res, indent=1), flush=True)
    return res


if __name__ == '__main__':
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    cfg = load_config(base.BASE / 'config.resolved.json'); records = load_manifest(base.BASE)
    rs = [run(s, cfg, records, torch.device('cuda')) for s in (42, 43, 44)]
    keys = ['auc_all_healthy_vs_all_fault_recordings', 'auc_clean_healthy_vs_healthy_quarter_in_fault_recordings', 'auc_within_1_2_fault_recordings']
    write_json(OUT / 'summary.json', {k: dict(seeds=[r[k] for r in rs], mean=float(np.mean([r[k] for r in rs if r[k] is not None]))) for k in keys})
    print({k: np.mean([r[k] for r in rs if r[k] is not None]) for k in keys})
