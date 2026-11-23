"""Unmix 3-speaker recordings into per-speaker spectra using mic directivity measured on 1-speaker recordings.
User request 2026-11-23 ("find usable features yourself").
Per recording: band power (64 mel bands + narrow 5 kHz line band) in 36 x 10-deg angle bins (time->angle by angle CSV).
Steering G[s](angle, band) = mean linear power profile of the 1-speaker HEALTHY recordings with only speaker s playing
(this includes the run-up time envelope and the room). Model for a 3-speaker recording:
    P(angle, band) ~= sum_s a_s(band) * G[s](angle, band)      a_s >= 0  (NNLS per band)
a_s(band) = estimated level of speaker s relative to a typical healthy file. Fault score of speaker s =
standardized distance of log a_s from the healthy reference (unmixed a_s of 0-fault recordings, same position,
leave-source-out). Labels only for evaluation. Compared with the plain sector feature (same bands, sector mean).
Output: runs/angle_profile_20261123/unmix_*.json, figures/unmix_1fault.png
"""
from pathlib import Path
import sys, json, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf
from scipy import signal
from scipy.optimize import nnls
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_manifest, read_json, data_path, write_json, load_config
from ev3dar.features import angle_track, mel_bank
ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
CFG = load_config(PROJECT / 'runs/baseline/config.resolved.json'); NB = 36; NF = 4096
W = mel_bank(dict(CFG, n_fft=NF)); FR = np.fft.rfftfreq(NF, 1 / 48000)
LINE = (FR >= 4990) & (FR <= 5010)
POS = ['FRONT', 'REAR', 'RIGHT']
SECT = {'FRONT': (135, 225), 'REAR': (315, 405), 'RIGHT': (225, 315)}


def prof(r):
    a, fs = sf.read(data_path(ROOT, r['wav']), dtype='float32'); at, aa = angle_track(data_path(ROOT, r['angles']), len(a) / fs)
    _, t, p = signal.spectrogram(a.astype(np.float64), fs=fs, window='hann', nperseg=NF, noverlap=NF - 480, scaling='spectrum', mode='psd')
    B = np.vstack([W @ p, p[LINE].mean(0, keepdims=True)])            # (65, frames)
    b = np.clip((np.interp(t, at, aa) // 10).astype(int), 0, NB - 1)
    return r['uid'], np.stack([B[:, b == i].mean(1) for i in range(NB)]).tolist()   # (36, 65) linear power


if __name__ == '__main__':
    out = PROJECT / 'runs/angle_profile_20261123'
    R1 = read_json(PROJECT / 'runs/ae_paper_single_20261123/manifest.json'); R3 = load_manifest(PROJECT / 'runs/baseline')
    cache = out / 'unmix_profiles.json'
    if cache.exists(): Pr = read_json(cache)
    else:
        with ProcessPoolExecutor(10) as ex: Pr = dict(ex.map(prof, R1 + R3, chunksize=8))
        write_json(cache, Pr)
    Pr = {k: np.array(v) for k, v in Pr.items()}
    G = {p: np.mean([Pr[r['uid']] for r in R1 if r['position'] == p and r['sound_class'] == 'Healthy'], 0) for p in POS}
    nb = G['FRONT'].shape[1]
    # steering condition number per band (how separable the 3 speakers are)
    cond = [float(np.linalg.cond(np.stack([G[p][:, j] / G[p][:, j].mean() for p in POS], 1))) for j in range(nb)]
    A, S = {}, {}
    for r in R3:
        P = Pr[r['uid']]; a = np.zeros((3, nb))
        for j in range(nb):
            M = np.stack([G[p][:, j] for p in POS], 1); sc = M.max()
            a[:, j], _ = nnls(M / sc, P[:, j] / sc)
        A[r['uid']] = np.log10(a + 1e-6) * 10                                # dB-like
        x = np.arange(NB) * 10 + 5
        S[r['uid']] = np.stack([10 * np.log10(P[((x >= lo) & (x < hi)) | ((x + 360 >= lo) & (x + 360 < hi))].mean(0) + 1e-20) for lo, hi in (SECT[p] for p in POS)])
    lk = {r['uid']: r for r in R3}; src = {r['uid']: set(r['source_files']) for r in R3}
    H0 = [r['uid'] for r in R3 if r['fault_count'] == 0]
    from sklearn.metrics import roc_auc_score
    res = dict(steering_condition_number=dict(median=float(np.median(cond)), band_5k_line=cond[-1]))
    for name, F in [('unmixed', A), ('sector', S)]:
        rows = []
        for r in R3:
            ref = [u for u in H0 if u != r['uid'] and not (src[u] & src[r['uid']])]
            for i, p in enumerate(POS):
                Hm = np.stack([F[u][i] for u in ref]); m, s = Hm.mean(0), Hm.std(0) + 1e-3; z = (F[r['uid']][i] - m) / s
                rows.append(dict(uid=r['uid'], pos=p, fc=r['fault_count'], y=int(r['classes'][i] != 'Healthy'), cls=r['classes'][i],
                                 score=float(np.sqrt(np.mean(z[:-1] ** 2))), line5k=float(z[-1])))
        for key in ['score', 'line5k']:
            s = np.array([x[key] for x in rows]); y = np.array([x['y'] for x in rows]); fc = np.array([x['fc'] for x in rows])
            pos = np.array([x['pos'] for x in rows]); uid = np.array([x['uid'] for x in rows]); cls = np.array([x['cls'] for x in rows])
            d = {}; m12 = (fc == 1) | (fc == 2)
            d['auc_within_1_2'] = roc_auc_score(y[m12], s[m12])
            for p in POS: k = m12 & (pos == p); d['auc_1_2_' + p] = roc_auc_score(y[k], s[k])
            top, fr, topc = [], [], collections.defaultdict(list)
            for u in np.unique(uid[fc == 1]):
                ii = np.where(uid == u)[0]; t = y[ii[np.argmax(s[ii])]]; top.append(t); topc[cls[ii][y[ii] == 1][0]].append(t)
                f = pos[ii][y[ii] == 1][0]
                if f != 'RIGHT':
                    o = 'REAR' if f == 'FRONT' else 'FRONT'; fr.append(s[ii][pos[ii] == f][0] > s[ii][pos[ii] == o][0])
            d['fc1_fault_ranked_first'] = float(np.mean(top)); d['fc1_front_vs_rear'] = float(np.mean(fr))
            d['fc1_ranked_first_by_class'] = {c: round(float(np.mean(v)), 2) for c, v in topc.items()}
            res[f'{name}_{key}'] = {k: (round(float(v), 3) if not isinstance(v, dict) else v) for k, v in d.items()}
    write_json(out / 'unmix_summary.json', res); print(json.dumps(res, indent=1))
