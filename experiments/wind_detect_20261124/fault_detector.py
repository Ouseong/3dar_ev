"""Combined narrow-feature fault detector on the 3DAR RECORDINGS (1200 three-speaker recordings), 2026-11-24.
Per recording three scores, each from one fault-specific component:
  Whine : max over the 3 sectors of the order-16 prominence (experiments/knowledge_rules_20261123, refined RPM)
  PWM   : max over the 3 sectors of the 5 kHz line prominence
  Wind  : logistic regression on the flank-detrended 3.3-6.2 kHz shape (mic facing 240-350 deg + remaining angles),
          trained on 1-/2-fault recordings of the training folds
Protocol: 5-fold cross-validation grouped by the RIGHT speaker's source file. In each fold the Wind model and the three
thresholds are fitted on the training folds only (threshold = best balanced accuracy for 'recording contains this fault'
vs 'does not'), then applied to the held-out fold. All numbers below are out-of-fold.
Reported: per-fault detection / false alarms (also split by recording block, to show robustness to the 11/09-morning
healthy block), any-fault decision, fault-type identification, PWM sector localisation.
Output: runs/wind_detect_20261124/detector_predictions.csv, detector_summary.json
"""
from pathlib import Path
import sys, json, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_manifest, write_json, write_csv
OUT = PROJECT / 'runs/wind_detect_20261124'
R = load_manifest(PROJECT / 'runs/baseline'); n = len(R); P3 = ['FRONT', 'REAR', 'RIGHT']
z = np.load(OUT / 'angle_spec.npz'); ix = {u: i for i, u in enumerate(z['uid'])}; G, EDG = z['G'], z['edges']; CEN = (EDG[:-1] + EDG[1:]) / 2; ANG = np.arange(36) * 10 + 5
fl = (CEN < 4000) | (CEN > 5500); Pm = np.vander(CEN / 1000, 2) @ np.linalg.pinv(np.vander(CEN[fl] / 1000, 2))
def shape(lo, hi): S = G[:, (ANG >= lo) & (ANG < hi)].mean(1); return S - S[:, fl] @ Pm.T
Xw_all = np.concatenate([shape(240, 350), shape(0, 240)], 1); Xw = Xw_all[[ix[r['uid']] for r in R]]
K = json.load(open(PROJECT / 'runs/knowledge_rules_20261123/roi_features_refined.json')); by = collections.defaultdict(dict)
for k in K: by[k['uid']][k['position']] = k
whine = np.array([max(by[r['uid']][p]['whine_o16'] for p in P3) for r in R]); pwm = np.array([max(by[r['uid']][p]['pwm_5k'] for p in P3) for r in R])
pwm_sec = np.array([[by[r['uid']][p]['pwm_5k'] for p in P3] for r in R])
has = {c: np.array([c in r['classes'] for r in R]) for c in ['Whine', 'PWM', 'Wind', 'Imbalance', 'Modulation']}
fc = np.array([r['fault_count'] for r in R]); date = np.array([r['event_time'][5:10] for r in R]); grp = np.array([r['source_files'][2] for r in R])


def best_thr(s, y):
    c = np.unique(s); ba = [((s[y] >= t).mean() + (s[~y] < t).mean()) / 2 for t in c]; return float(c[int(np.argmax(ba))])


score = {'Whine': whine, 'PWM': pwm, 'Wind': np.zeros(n)}; pred = {c: np.zeros(n, bool) for c in score}
for tr, te in GroupKFold(5).split(np.arange(n), has['Wind'], grp):
    m12 = tr[(fc[tr] == 1) | (fc[tr] == 2)]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000, class_weight='balanced')).fit(Xw[m12], has['Wind'][m12])
    score['Wind'][te] = clf.decision_function(Xw[te]); wtr = clf.decision_function(Xw[tr])
    for c, s_tr in [('Whine', whine[tr]), ('PWM', pwm[tr]), ('Wind', wtr)]:
        k = fc[tr] > 0                                         # thresholds from fault-block recordings only (same recording conditions)
        pred[c][te] = score[c][te] >= best_thr(s_tr[k], has[c][tr][k])
S = {}
print('%-6s %8s %10s | false alarms: %-12s %-12s %-22s' % ('fault', 'detected', '(n)', 'healthy 11/09', 'healthy 11/02', 'fault rec. without it'))
for c in score:
    d = dict(detected=float(pred[c][has[c]].mean()), n=int(has[c].sum()), fa_healthy_1109=float(pred[c][(fc == 0) & (date == '11-09')].mean()),
             fa_healthy_1102=float(pred[c][(fc == 0) & (date == '11-02')].mean()), fa_fault_rec_without=float(pred[c][(fc > 0) & ~has[c]].mean()),
             auc_all=float(roc_auc_score(has[c], score[c])))
    for f in (1, 2, 3):
        k = (fc == f) & has[c]; d[f'detected_fc{f}'] = float(pred[c][k].mean()) if k.any() else None
    S[c] = d; print('%-6s %8.2f %10d | %21.2f %12.2f %22.2f   detected by fault count 1/2/3: %s' % (c, d['detected'], d['n'], d['fa_healthy_1109'], d['fa_healthy_1102'], d['fa_fault_rec_without'],
                    '/'.join('%.2f' % d[f'detected_fc{f}'] if d[f'detected_fc{f}'] is not None else '-' for f in (1, 2, 3))))
anyp = pred['Whine'] | pred['PWM'] | pred['Wind']; det = has['Whine'] | has['PWM'] | has['Wind']
S['any_fault'] = dict(recall_recordings_with_whine_pwm_wind=float(anyp[det].mean()), n=int(det.sum()),
                      flagged_all_healthy_1109=float(anyp[(fc == 0) & (date == '11-09')].mean()), flagged_all_healthy_1102=float(anyp[(fc == 0) & (date == '11-02')].mean()),
                      flagged_only_imbalance_or_modulation=float(anyp[(fc > 0) & ~det].mean()), n_only_imb_mod=int(((fc > 0) & ~det).sum()),
                      accuracy_healthy_vs_wpw=float(np.r_[anyp[det], ~anyp[fc == 0]].mean()))
print('\nany of Whine/PWM/Wind:', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in S['any_fault'].items()})
# type identification on 1-fault recordings with Whine/PWM/Wind
k1 = (fc == 1) & det; true = np.array([[c for c in r['classes'] if c != 'Healthy'][0] if r['fault_count'] == 1 else '' for r in R])
exact = np.array([set(c for c in score if pred[c][i]) == {true[i]} for i in np.where(k1)[0]])
S['type_id_1fault'] = dict(n=int(k1.sum()), exactly_the_right_type=float(exact.mean()),
                           per_type={c: float(exact[true[k1] == c].mean()) for c in score})
print('1-fault recordings (Whine/PWM/Wind): exactly the right type flagged', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in S['type_id_1fault'].items()})
# multi-label exact match over all recordings (only W/P/Wind labels)
ex = np.array([all(pred[c][i] == has[c][i] for c in score) for i in range(n)])
S['exact_set_all_recordings'] = {f'fc{f}': float(ex[fc == f].mean()) for f in range(4)}; print('exact set of {Whine, PWM, Wind} correct:', {k: round(v, 2) for k, v in S['exact_set_all_recordings'].items()})
# PWM localisation when PWM flagged and exactly one PWM speaker
loc = [(P3[int(np.argmax(pwm_sec[i]))] == P3[R[i]['classes'].index('PWM')]) for i in range(n) if pred['PWM'][i] and R[i]['classes'].count('PWM') == 1]
S['pwm_localisation'] = dict(n=len(loc), correct_sector=float(np.mean(loc))); print('PWM sector (flagged, one PWM speaker):', S['pwm_localisation'])
write_json(OUT / 'detector_summary.json', S)
write_csv(OUT / 'detector_predictions.csv', [dict(uid=r['uid'], date=date[i], fault_count=int(fc[i]), classes='|'.join(r['classes']),
          **{f'score_{c}': float(score[c][i]) for c in score}, **{f'pred_{c}': int(pred[c][i]) for c in score}) for i, r in enumerate(R)])
