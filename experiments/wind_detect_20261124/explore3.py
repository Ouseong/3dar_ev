"""Third round: (1) Wind models on the full angle x frequency map, more training data (all 1200 recordings) but still
scored inside (date, fault-count) blocks; (2) the same in-block test for PWM and Whine with the knowledge features
(5 kHz line, order-16 prominence; max over the three sectors)."""
from pathlib import Path
import sys, json, collections
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(Path(__file__).parent))
import explore as E
G, CEN, ANG, ix, blocks, R3 = E.G, E.CEN, E.ANG, E.ix, E.blocks, E.R3
lk = {r['uid']: r for r in R3}; fl = (CEN < 4000) | (CEN > 5500)
V = np.vander(CEN[fl] / 1000, 2); Pm = np.vander(CEN / 1000, 2) @ np.linalg.pinv(V)          # flank line fit as a matrix
def detr(S): return S - S[..., fl] @ Pm.T
def shape(lo, hi): return detr(G[:, (ANG >= lo) & (ANG < hi)].mean(1))
six = lambda: np.stack([detr(G[:, (ANG >= a) & (ANG < a + 60)].mean(1)) for a in range(0, 360, 60)], 1).reshape(len(G), -1)
feats = {'right 240-350 (58)': shape(240, 350), 'right + other (116)': np.concatenate([shape(240, 350), shape(0, 240)], 1), '6 angle sectors (348)': six()}
allr = np.array([ix[r['uid']] for r in R3]); y_all = np.array([int(r['classes'][2] == 'Wind') for r in R3]); g_all = np.array([r['source_files'][2] for r in R3])
inb = np.array([i for b in blocks.values() for i in b[0] + b[1]]); sel_in = np.isin(allr, inb)
models = {'LR': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight='balanced')),
          'PCA20+LR': lambda: make_pipeline(StandardScaler(), PCA(20, random_state=0), LogisticRegression(C=1, max_iter=5000, class_weight='balanced')),
          'GBM': lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05, class_weight='balanced', random_state=0)}
print('Wind: in-block AUC [11/02 fc1, 11/02 fc2, 11/09 fc1]')
best = None
for fn, X in feats.items():
    for mn, mk in models.items():
        for train_name, tsel in [('train fc1-2', sel_in), ('train all', np.ones(len(allr), bool))]:
            s = np.full(len(G), np.nan)
            for tr, te in GroupKFold(5).split(allr, y_all, g_all):
                tr = tr[tsel[tr]]; m = mk().fit(X[allr[tr]], y_all[tr])
                s[allr[te]] = m.predict_proba(X[allr[te]])[:, 1]
            s[E.w1 + E.h1] = 0.0; a, per, _ = E.ev(s); print('  %-22s %-9s %-11s %.3f %s' % (fn, mn, train_name, a, [round(x, 2) for x in per]))
            if best is None or a > best[0]: best = (a, fn, mn, train_name, s.copy())
print('best:', best[:4])
s = best[4]; P = np.array([i for b in blocks.values() for i in b[0]]); N = np.array([i for b in blocks.values() for i in b[1]])
for fa in (0.05, 0.10, 0.20):
    thr = np.quantile(s[N], 1 - fa); print('  at %.0f%% false alarms: Wind detected %.2f' % (fa * 100, (s[P] > thr).mean()))
np.save(PROJECT / 'runs/wind_detect_20261124/wind_score_best.npy', s)
# ---- PWM and Whine, same in-block protocol ----
K = json.load(open(PROJECT / 'runs/knowledge_rules_20261123/roi_features_refined.json')); by = collections.defaultdict(dict)
for r in K: by[r['uid']][r['position']] = r
print('\nPWM / Whine recording-level detection inside blocks (max over sectors of the knowledge feature):')
for cls, feat in [('PWM', 'pwm_5k'), ('Whine', 'whine_o16')]:
    num = den = 0; per = {}
    bl = collections.defaultdict(lambda: ([], []))
    for r in R3:
        if r['fault_count'] in (1, 2): bl[(r['event_time'][:10], r['fault_count'])][0 if cls in r['classes'] else 1].append(max(by[r['uid']][p][feat] for p in by[r['uid']]))
    for b, (p, n) in sorted(bl.items()):
        if p and n: a = roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n]); per[f'{b[0][5:]} fc{b[1]}'] = (round(float(a), 2), len(p), len(n)); num += a * len(p) * len(n); den += len(p) * len(n)
    print('  %-6s %-10s in-block AUC %.3f  %s' % (cls, feat, num / den, per))
