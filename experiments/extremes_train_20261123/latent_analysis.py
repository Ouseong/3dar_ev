"""Latent-space analysis of the model trained on 0-fault / 3-fault recordings only (user request 2026-11-23).
Loads the definitions of run_binary_extremes.py (everything before the multi-seed loop; no training), the seed-42
checkpoint, and extracts the 64-d latent for every quarter of all 1200 recordings (on GPU, inference only).
Analyses:
  1. PCA / t-SNE coloured by recording fault count and quarter label   -> figures/latent_extremes_seed42.png
  2. How much latent variance is explained by recording identity, position, date, quarter label
  3. Linear probes (logistic regression, recording-grouped 5-fold CV) on 1-/2-fault recordings that were NOT used
     for training: fault-vs-healthy quarter from (a) raw latent, (b) latent minus the recording mean,
     plus position and recording date from the raw latent.
"""
from pathlib import Path
import sys, json
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / 'experiments/extremes_train_20261123/run_binary_extremes.py'
code = SRC.read_text(encoding='utf-8'); code = code[:code.index('# 8. Multi-Seed Loop Start!')]
G = {'__name__': 'latent_defs'}; exec(compile(code, str(SRC), 'exec'), G)

seed = 42
ck = torch.load(PROJECT / f'runs/nb_binary_spec_extremes_20261123/seed{seed}/id_best.pt', map_location='cpu', weights_only=False)
model = G['build_model']('id', ck['config']); model.load_state_dict(ck['state_dict']); model.eval().cuda()
records = G['discover'](G['CONFIG']['data_root']); split = ck['split']
role = {u: k for k in ['train', 'validation', 'test'] for u in split[k]}
loader = G['make_loader'](records, ck['normalization'], None)
Z, meta = [], []
lk = {r['uid']: r for r in records}
with torch.no_grad():
    for b in loader:
        _, lat = G['forward'](model, 'id', b, torch.device('cuda'))          # (B, 3, 64)
        for j, u in enumerate(b['uid']):
            r = lk[u]
            for k, pos in enumerate(G['POS']):
                Z.append(lat[j, k].cpu().numpy())
                meta.append(dict(uid=u, pos=pos, fc=r['fault_count'], fault=int(r['classes'][k] != 'Healthy'),
                                 cls=r['classes'][k], date=r['event_time'][:10], role=role.get(u, 'dropped')))
Z = np.array(Z); uid = np.array([m['uid'] for m in meta]); fc = np.array([m['fc'] for m in meta])
y = np.array([m['fault'] for m in meta]); pos = np.array([m['pos'] for m in meta]); date = np.array([m['date'] for m in meta])
roleA = np.array([m['role'] for m in meta])
out = {}

# 2. variance decomposition (fraction of total latent variance explained by group means)
def explained(groups, mask=None):
    m = np.ones(len(Z), bool) if mask is None else mask; X = Z[m]; g = groups[m]; tot = ((X - X.mean(0)) ** 2).sum()
    between = sum((g == v).sum() * ((X[g == v].mean(0) - X.mean(0)) ** 2).sum() for v in np.unique(g))
    return float(between / tot)
mid = (fc == 1) | (fc == 2)
out['variance_explained_in_1_2_fault_recordings'] = dict(
    recording_identity=explained(uid, mid), position=explained(pos, mid), date=explained(date, mid),
    quarter_label=explained(y.astype(str), mid), fault_count=explained(fc.astype(str), mid))
out['variance_explained_all'] = dict(recording_identity=explained(uid), fault_count=explained(fc.astype(str)),
                                     date=explained(date), position=explained(pos))

# 3. probes on 1/2-fault recordings not used for training (test + dropped)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
m = mid & (roleA != 'train') & (roleA != 'validation')
Zc = Z.copy()
for u in np.unique(uid): k = uid == u; Zc[k] = Z[k] - Z[k].mean(0)       # remove recording-level component
clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
gkf = GroupKFold(5)
probe = {}
for name, X in [('raw_latent', Z), ('latent_minus_recording_mean', Zc)]:
    p = cross_val_predict(clf(), X[m], y[m], groups=uid[m], cv=gkf, method='predict_proba')[:, 1]
    probe['fault_vs_healthy_' + name] = dict(auc=float(roc_auc_score(y[m], p)), n=int(m.sum()))
    # within 1-fault recordings: is the fault quarter ranked first?
    m1 = np.where(m & (fc == 1))[0]; pm = dict(zip(m.nonzero()[0], p)); top = []
    for u in np.unique(uid[m1]):
        ii = [i for i in m1 if uid[i] == u]; top.append(int(y[max(ii, key=lambda i: pm[i])]))
    probe['fault_vs_healthy_' + name]['fault_quarter_ranked_first_1fault'] = float(np.mean(top))
    probe['fault_vs_healthy_' + name]['n_1fault_recordings'] = len(top)
for name, lab in [('position', pos), ('date', date)]:
    pr = cross_val_predict(clf(), Z[m], lab[m], groups=uid[m], cv=gkf)
    probe[name + '_from_raw_latent'] = dict(accuracy=float(accuracy_score(lab[m], pr)),
                                           chance=float(max(np.mean(lab[m] == v) for v in np.unique(lab[m]))))
out['linear_probes_on_unseen_1_2_fault_recordings'] = probe
# model's own output on the same set
out['n_quarters'] = int(len(Z))
(PROJECT / 'runs/nb_binary_spec_extremes_20261123/latent_analysis_seed42.json').write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))

# 1. figures
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rng = np.random.default_rng(0); sub = rng.choice(len(Z), 2400, replace=False)
pc = PCA(2).fit_transform(StandardScaler().fit_transform(Z))
ts = TSNE(2, perplexity=40, init='pca', random_state=0).fit_transform(StandardScaler().fit_transform(Z[sub]))
fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
cols = {0: '#2ca02c', 1: '#ffbf00', 2: '#ff7f0e', 3: '#d62728'}
for f in range(4):
    k = fc == f; ax[0].scatter(pc[k, 0], pc[k, 1], s=4, c=cols[f], alpha=.5, label=f'{f}-fault recording')
ax[0].set_title('PCA of latent (all 3600 quarters), colour = recording fault count'); ax[0].legend(markerscale=4)
for f in range(4):
    k = fc[sub] == f; ax[1].scatter(ts[k, 0], ts[k, 1], s=5, c=cols[f], alpha=.6, label=f'{f}-fault rec.')
ax[1].set_title('t-SNE (2400 quarters), colour = recording fault count'); ax[1].legend(markerscale=4)
k1 = (fc[sub] == 1) | (fc[sub] == 2)
ax[2].scatter(ts[~k1, 0], ts[~k1, 1], s=4, c='0.85')
for lab, col, nm in [(0, '#1f77b4', 'healthy quarter'), (1, '#d62728', 'fault quarter')]:
    k = k1 & (y[sub] == lab); ax[2].scatter(ts[k, 0], ts[k, 1], s=7, c=col, alpha=.7, label=nm + ' (1/2-fault rec.)')
ax[2].set_title('t-SNE, only 1-/2-fault recordings: quarter label'); ax[2].legend(markerscale=3)
plt.tight_layout(); plt.savefig(PROJECT / 'figures/latent_extremes_seed42.png', dpi=90)
