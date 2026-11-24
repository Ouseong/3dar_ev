"""Second round: wider angle windows, right-minus-reference contrast, learned template (logistic regression on the
flank-detrended 3.3-6.2 kHz shape, CV grouped by the RIGHT speaker's source file), and how the recorded bump relates to
the bump size in the original Wind file that was played."""
from pathlib import Path
import sys, json, collections, csv
import numpy as np, soundfile as sf
from scipy import signal
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(Path(__file__).parent))
import explore as E
from ev3dar.common import data_path
G, CEN, ANG, ix, blocks = E.G, E.CEN, E.ANG, E.ix, E.blocks
def gspec(lo, hi): return G[:, (ANG >= lo) & (ANG < hi)].mean(1)
core, f1, f2 = (4500, 4900), (3700, 4000), (5500, 5800)
print('angle windows (G mean):')
for lo, hi in [(250, 310), (230, 360), (250, 360), (270, 360), (240, 340), (260, 350), (0, 360)]:
    a, per, o = E.ev(E.bump(gspec(lo, hi), core, f1, f2)); print('  %3d-%3d  in-block %.3f %s  1spk %.2f' % (lo, hi, a, [round(x, 2) for x in per], o))
b_r = E.bump(gspec(240, 350), core, f1, f2)
for name, (lo, hi) in [('front 135-225', (135, 225)), ('empty 45-135', (45, 135)), ('0-200', (0, 200))]:
    a, per, o = E.ev(b_r - E.bump(gspec(lo, hi), core, f1, f2)); print('contrast right(240-350) minus %-14s in-block %.3f %s 1spk %.2f' % (name, a, [round(x, 2) for x in per], o))
# learned template
R3 = E.R3; lk = {r['uid']: r for r in R3}
use = np.array([i for b in blocks.values() for i in b[0] + b[1]]); y = np.array([int(lk[E.UID[i]]['classes'][2] == 'Wind') for i in use])
grp = np.array([lk[E.UID[i]]['source_files'][2] for i in use])
def shape(lo, hi):
    S = gspec(lo, hi); fl = (CEN < 4000) | (CEN > 5500); out = np.zeros_like(S)
    for i in range(len(S)): out[i] = S[i] - np.polyval(np.polyfit(CEN[fl], S[i][fl], 1), CEN)
    return out
for desc, X in [('shape right 240-350', shape(240, 350)), ('shape right + shape other', np.concatenate([shape(240, 350), shape(0, 240)], 1))]:
    s = np.full(len(G), np.nan)
    for tr, te in GroupKFold(5).split(use, y, grp):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced')).fit(X[use[tr]], y[tr]); s[use[te]] = clf.decision_function(X[use[te]])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight='balanced')).fit(X[use], y)
    s[E.w1 + E.h1] = clf.decision_function(X[E.w1 + E.h1])
    a, per, o = E.ev(s); print('learned template LR (%s): in-block %.3f %s 1spk(transfer) %.2f' % (desc, a, [round(x, 2) for x in per], o))
# relation to the original Wind file's bump size
KU = Path('/home/kos/Desktop/3DAR_EV/KU_Leuven_original'); ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
FR = np.fft.rfftfreq(4096, 1 / 48000)
def orig_bump(r):
    row = [x for x in csv.DictReader(open(data_path(ROOT, r['info']), encoding='utf-8-sig')) if x['Speaker_Loc'] == 'RIGHT'][0]
    t0, t1 = map(float, row['Source_Interval_Sec'].split('~')); name = row['Source_File'].replace('\\', '/').split('/')[-1]
    s, fs = sf.read(KU / row['Sound_Class'] / name, start=round(t0 * 48000), stop=round(t1 * 48000), dtype='float32')
    _, p = signal.welch(s.astype(np.float64), fs, nperseg=4096); db = 10 * np.log10(p + 1e-30); lv = lambda a, b: db[(FR >= a) & (FR < b) & ~((FR > 4975) & (FR < 5025))].mean()
    return lv(*core) - .5 * (lv(*f1) + lv(*f2))
P = [i for b in blocks.values() for i in b[0]]; N = [i for b in blocks.values() for i in b[1]]
ob = np.array([orig_bump(lk[E.UID[i]]) for i in P]); on = np.array([orig_bump(lk[E.UID[i]]) for i in N[:150]])
print('original bump of the played RIGHT file: Wind median %.1f dB [p10 %.1f, p90 %.1f]; Healthy mean %.2f std %.2f' % (np.median(ob), *np.percentile(ob, [10, 90]), on.mean(), on.std()))
print('corr(recorded bump facing right, original bump) among Wind recordings: %.2f' % np.corrcoef(b_r[P], ob)[0, 1])
thr = np.percentile(b_r[N], 90)
for lo, hi in [(0, 2), (2, 3), (3, 4), (4, 9)]:
    k = (ob >= lo) & (ob < hi); print('  original bump %d-%d dB: n=%3d  detected at 10%% false-alarm threshold: %.2f  mean recorded bump %.1f (no-Wind mean %.1f)' % (lo, hi, k.sum(), (b_r[P][k] > thr).mean() if k.sum() else np.nan, b_r[P][k].mean() if k.sum() else np.nan, b_r[N].mean()))
