"""Wind-fault detection features on the 3DAR recordings (user request: 'devise a way to catch faults up to Wind').
Original files show the Wind fault as a spectral bump at 4.0-5.5 kHz (peak 4.5-5.0 kHz, +4..6.5 dB at 4-8k RPM,
per-file 0.5..6.8 dB), nothing else changes. Features (dB, self-normalised inside each recording, PWM 5 kHz line excluded):
  bump_wide   : mean(3.9-5.6k) - mean of flanks (2.5-3.5k, 6.5-7.5k)            [old feature]
  bump_narrow : mean(4.4-4.95k) - mean of flanks (3.5-3.9k, 5.6-6.0k)
  bump_tmpl   : log-spectrum 3.4-6.1 kHz (50 Hz bins), straight line through the two flanks removed, projected on the
                bump template measured from the ORIGINAL files (mean Wind - mean Healthy, unit norm)
each computed on (whole) the whole recording, (right) only while the mic faces RIGHT 225-315 deg, (other) the rest,
and contrast = right - other.
Evaluation without the recording-block confound: positives = recordings whose RIGHT speaker plays Wind, negatives =
recordings whose RIGHT speaker is Healthy, compared only inside the same (date, fault_count) block (fc 1 and 2), AUC
averaged over blocks weighted by pair count. Also 1-speaker RIGHT: Wind (20) vs Healthy (20).
Output: runs/wind_detect_20261124/{features.json, summary.json}
"""
from pathlib import Path
import sys, json, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf
from scipy import signal
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_manifest, read_json, write_json, data_path
from ev3dar.features import angle_track
ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV'); KU = Path('/home/kos/Desktop/3DAR_EV/KU_Leuven_original')
OUT = PROJECT / 'runs/wind_detect_20261124'; NF = 4096
FR = np.fft.rfftfreq(NF, 1 / 48000); EDG = np.arange(3400, 6101, 50); CEN = (EDG[:-1] + EDG[1:]) / 2
NOT_PWM = ~((FR > 4975) & (FR < 5025))


def template():
    FF = read_json(PROJECT / 'runs/ku_original_20261123/file_features.json'); rng = np.random.default_rng(0)
    def spec(f):
        a, fs = sf.read(KU / f['cls'] / f['name'], dtype='float32'); i0, i1 = int(.5 * len(a)), int(.8 * len(a))
        _, p = signal.welch(a[i0:i1].astype(np.float64), fs, nperseg=NF); return 10 * np.log10(p + 1e-30)
    H = [f for f in FF if f['cls'] == 'Healthy']; Wd = [f for f in FF if f['cls'] == 'Wind']
    d = np.mean([spec(f) for f in Wd], 0) - np.mean([spec(H[i]) for i in rng.choice(len(H), 200, replace=False)], 0)
    t = np.array([d[(FR >= a) & (FR < b) & NOT_PWM].mean() for a, b in zip(EDG[:-1], EDG[1:])]); t = np.maximum(t, 0)
    return (t / np.linalg.norm(t)).tolist()


def feats(db, tmpl):
    lv = lambda a, b: db[(FR >= a) & (FR < b) & NOT_PWM].mean()
    s = np.array([lv(a, b) for a, b in zip(EDG[:-1], EDG[1:])])
    fl = (CEN < 3900) | (CEN > 5600); co = np.polyfit(CEN[fl], s[fl], 1); r = s - np.polyval(co, CEN)
    return dict(bump_wide=lv(3900, 5600) - .5 * (lv(2500, 3500) + lv(6500, 7500)), bump_narrow=lv(4400, 4950) - .5 * (lv(3500, 3900) + lv(5600, 6000)),
                bump_tmpl=float(r @ np.array(tmpl)))


def one(args):
    r, tmpl = args
    a, fs = sf.read(data_path(ROOT, r['wav']), dtype='float32'); at, aa = angle_track(data_path(ROOT, r['angles']), len(a) / fs)
    f, t, p = signal.spectrogram(a.astype(np.float64), fs=fs, window='hann', nperseg=NF, noverlap=NF - 480, scaling='spectrum', mode='psd')
    ang = np.interp(t, at, aa); right = (ang >= 225) & (ang < 315)
    out = dict(uid=r['uid'])
    for tag, m in [('whole', np.ones(len(t), bool)), ('right', right), ('other', ~right)]:
        for k, v in feats(10 * np.log10(p[:, m].mean(1) + 1e-30), tmpl).items(): out[f'{k}_{tag}'] = float(v)
    for k in ('bump_wide', 'bump_narrow', 'bump_tmpl'): out[f'{k}_contrast'] = out[f'{k}_right'] - out[f'{k}_other']
    return out


def auc(pos, neg):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True); tmpl = template()
    R3 = load_manifest(PROJECT / 'runs/baseline'); R1 = read_json(PROJECT / 'runs/ae_paper_single_20261123/manifest.json')
    with ProcessPoolExecutor(12) as ex: F = list(ex.map(one, [(r, tmpl) for r in R3 + R1], chunksize=8))
    F = {f['uid']: f for f in F}; write_json(OUT / 'features.json', dict(template=tmpl, features=F))
    keys = [k for k in next(iter(F.values())) if k != 'uid']; S = {}
    blocks = collections.defaultdict(lambda: ([], []))
    for r in R3:
        if r['fault_count'] in (1, 2): blocks[(r['event_time'][:10], r['fault_count'])][0 if r['classes'][2] == 'Wind' else 1].append(r['uid'])
    S['blocks'] = {f'{k[0]} fc{k[1]}': dict(wind=len(v[0]), no_wind=len(v[1])) for k, v in sorted(blocks.items())}
    w1 = [r['uid'] for r in R1 if r['sound_class'] == 'Wind']; h1 = [r['uid'] for r in R1 if r['sound_class'] == 'Healthy' and r['position'] == 'RIGHT']
    h0 = [r['uid'] for r in R3 if r['fault_count'] == 0]; wall = [r['uid'] for r in R3 if r['classes'][2] == 'Wind']
    for k in keys:
        num = den = 0; per = {}
        for b, (p, n) in sorted(blocks.items()):
            if p and n: a = auc([F[u][k] for u in p], [F[u][k] for u in n]); per[f'{b[0][5:]} fc{b[1]}'] = round(float(a), 3); num += a * len(p) * len(n); den += len(p) * len(n)
        S[k] = dict(within_block_auc=round(float(num / den), 3), per_block=per, one_speaker_right_auc=round(float(auc([F[u][k] for u in w1], [F[u][k] for u in h1])), 3),
                    confounded_wind_vs_all_healthy_rec=round(float(auc([F[u][k] for u in wall], [F[u][k] for u in h0])), 3),
                    mean_wind=round(float(np.mean([F[u][k] for u in wall])), 2), mean_no_wind_fault_rec=round(float(np.mean([F[u][k] for b in blocks.values() for u in b[1]])), 2))
    write_json(OUT / 'summary.json', S)
    print('blocks', S['blocks'])
    print('%-22s %8s %8s %8s   %s' % ('feature', 'in-block', '1spk', 'confnd', 'per block'))
    for k in keys: print('%-22s %8.3f %8.3f %8.3f   %s' % (k, S[k]['within_block_auc'], S[k]['one_speaker_right_auc'], S[k]['confounded_wind_vs_all_healthy_rec'], S[k]['per_block']))
