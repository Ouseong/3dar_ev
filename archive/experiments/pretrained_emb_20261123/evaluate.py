"""Supervised and unsupervised Healthy/Fault scoring on BEATs / YAMNet ROI embeddings (user request 2026-11-23).
ROI embedding = duration-weighted mean of its view embeddings (REAR has 2 views).
SUPERVISED: strict source-disjoint splits runs/strict_split_20261123/seed{42,43,44}. StandardScaler + logistic regression
  (class_weight balanced); C chosen from {0.001, 0.01, 0.1, 1} by validation AUC. Same metrics as the CNN strict run.
UNSUPERVISED (no fault labels): healthy reference = ROIs of 0-fault recordings. 0-fault recordings are split into 5 folds
  by source-sharing components; for every fold a model (StandardScaler -> PCA 32 -> Ledoit-Wolf Mahalanobis, per position)
  is fit on the other folds. A ROI is scored by the fold models whose reference shares no source file with its recording
  (held-out 0-fault ROIs by their own fold model). Score = Mahalanobis distance.
Output: runs/pretrained_emb_20261123/summary.json
"""
from pathlib import Path
import sys, json, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_manifest, read_json, write_json
OUT = PROJECT / 'runs/pretrained_emb_20261123'
R = load_manifest(PROJECT / 'runs/baseline'); lk = {r['uid']: r for r in R}; P3 = ['FRONT', 'REAR', 'RIGHT']
z = np.load(OUT / 'segments_16k.npz', allow_pickle=True)
emb_views = {'BEATs': np.load(OUT / 'beats_views.npy'), 'YAMNet': np.load(OUT / 'yamnet_views.npy')}
keys = sorted({(u, p) for u, p in zip(z['uid'], z['position'])}, key=lambda k: (k[0], P3.index(k[1])))
kidx = {k: i for i, k in enumerate(keys)}
def roi_emb(E):
    S = np.zeros((len(keys), E.shape[1])); W = np.zeros(len(keys))
    for e, u, p, d in zip(E, z['uid'], z['position'], z['duration']): i = kidx[(u, p)]; S[i] += e * d; W[i] += d
    return S / W[:, None]
EMB = {k: roi_emb(v) for k, v in emb_views.items()}; EMB['BEATs+YAMNet'] = np.concatenate([EMB['BEATs'], EMB['YAMNet']], 1)
uid = np.array([k[0] for k in keys]); pos = np.array([k[1] for k in keys])
cls = np.array([lk[u]['classes'][P3.index(p)] for u, p in keys]); y = (cls != 'Healthy').astype(int); fc = np.array([lk[u]['fault_count'] for u in uid])


def metrics(s, m):
    s, yy, ff, uu, cc = s[m], y[m], fc[m], uid[m], cls[m]
    res = dict(auc_all=roc_auc_score(yy, s))
    k = (yy == 1) | (ff > 0); res['auc_hard_healthy'] = roc_auc_score(yy[k], s[k])
    k = (ff == 1) | (ff == 2); res['auc_within_1_2'] = roc_auc_score(yy[k], s[k]) if k.sum() and 0 < yy[k].mean() < 1 else np.nan
    k = (ff == 0) | (ff == 3); res['auc_all0_vs_all3'] = roc_auc_score(ff[k] == 3, s[k]) if (ff[k] == 3).any() and (ff[k] == 0).any() else np.nan
    k = (ff == 0) | ((ff > 0) & (yy == 0)); res['auc_clean_vs_healthy_in_fault_rec'] = roc_auc_score(ff[k] > 0, s[k])
    top = [yy[uu == u][np.argmax(s[uu == u])] for u in np.unique(uu[ff == 1])]; res['fc1_fault_ranked_first'] = float(np.mean(top))
    return res


summary = {}
for name, X in EMB.items():
    # ---------- supervised ----------
    per = collections.defaultdict(list)
    for seed in (42, 43, 44):
        sp = read_json(PROJECT / f'runs/strict_split_20261123/seed{seed}.json')
        tr, va, te = (np.isin(uid, sp[k]) for k in ('train', 'validation', 'test'))
        sc = StandardScaler().fit(X[tr]); best = None
        for C in (0.001, 0.01, 0.1, 1):
            clf = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(sc.transform(X[tr]), y[tr])
            a = roc_auc_score(y[va], clf.predict_proba(sc.transform(X[va]))[:, 1])
            if best is None or a > best[0]: best = (a, C, clf)
        p = best[2].predict_proba(sc.transform(X))[:, 1]; res = metrics(p, te); pred = p >= .5
        res['accuracy'] = float((pred[te] == y[te]).mean()); res['C'] = best[1]
        for c in ['Imbalance', 'Modulation', 'Whine', 'PWM', 'Wind']: res['recall_' + c] = float(pred[te & (cls == c)].mean())
        for k, v in res.items(): per[k].append(v)
    summary[name + ' | supervised'] = {k: round(float(np.nanmean(v)), 3) for k, v in per.items()}
    # ---------- unsupervised ----------
    h0 = sorted({u for u in uid if lk[u]['fault_count'] == 0})
    par = {u: u for u in h0}
    def f(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    first = {}
    for u in h0:
        for s_ in lk[u]['source_files']:
            if s_ in first: par[f(u)] = f(first[s_])
            else: first[s_] = u
    comps = collections.defaultdict(list)
    for u in h0: comps[f(u)].append(u)
    rng = np.random.default_rng(0); groups = list(comps.values()); rng.shuffle(groups); groups.sort(key=len, reverse=True)
    folds = [[] for _ in range(5)]
    for g in groups: min(folds, key=len).extend(g)
    S = np.zeros(len(keys)); N = np.zeros(len(keys))
    for k in range(5):
        ref = set(u for j in range(5) if j != k for u in folds[j]); refsrc = {s_ for u in ref for s_ in lk[u]['source_files']}
        target = np.array([(u in folds[k]) or (lk[u]['fault_count'] > 0 and not (set(lk[u]['source_files']) & refsrc)) for u in uid])
        for p_ in P3:
            mref = np.isin(uid, list(ref)) & (pos == p_)
            sc = StandardScaler().fit(X[mref]); pca = PCA(32, random_state=0).fit(sc.transform(X[mref]))
            lw = LedoitWolf().fit(pca.transform(sc.transform(X[mref])))
            mt = target & (pos == p_)
            S[mt] += np.sqrt(lw.mahalanobis(pca.transform(sc.transform(X[mt])))); N[mt] += 1
    ok = N > 0; s = np.where(ok, S / np.maximum(N, 1), np.nan)
    res = metrics(np.nan_to_num(s), ok); res['n_scored_rois'] = int(ok.sum())
    res.update({'fault_count_scored_' + str(k): int((ok & (fc == k)).sum()) for k in range(4)})
    summary[name + ' | unsupervised'] = {k: round(float(v), 3) for k, v in res.items()}
write_json(OUT / 'summary.json', summary)
for k, v in summary.items(): print('==', k); print('   ', v)
