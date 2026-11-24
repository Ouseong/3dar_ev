"""Band level vs microphone angle (10 deg bins), 3-speaker recordings (user request 2026-11-23).
For each recording: STFT (n_fft 4096, hop 480), frame time -> angle via the angle CSV, band power averaged per 10 deg bin.
Pattern of 1-fault recordings (grouped by fault position & class) minus mean pattern of 0-fault recordings, so that
effects common to all recordings (RPM run-up over time, room, speaker layout) cancel. A directional fault should
appear as a bump at its speaker direction (assumed FRONT 180, RIGHT 270, REAR 0 deg).
Output: figures/angle_profile_1fault.png, runs/angle_profile_20261123/profiles.json
"""
from pathlib import Path
import sys, json, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf
from scipy import signal
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_manifest, data_path, write_json
from ev3dar.features import angle_track
ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
BANDS = {'0.3-0.7k': (300, 700), '0.7-1.5k': (700, 1500), '1.5-2.5k': (1500, 2500), '3.75-5.75k (Wind)': (3750, 5750), '5k line (PWM)': (4990, 5010)}
NB = 36


def prof(r):
    a, fs = sf.read(data_path(ROOT, r['wav']), dtype='float32'); dur = len(a) / fs; at, aa = angle_track(data_path(ROOT, r['angles']), dur)
    f, t, p = signal.spectrogram(a.astype(np.float64), fs=fs, window='hann', nperseg=4096, noverlap=4096 - 480, scaling='spectrum', mode='psd')
    ang = np.interp(t, at, aa); b = np.clip((ang // (360 / NB)).astype(int), 0, NB - 1)
    out = {}
    for k, (lo, hi) in BANDS.items():
        e = p[(f >= lo) & (f < hi)].mean(0)
        out[k] = [float(10 * np.log10(e[b == i].mean() + 1e-20)) if (b == i).any() else np.nan for i in range(NB)]
    return r['uid'], out


if __name__ == '__main__':
    R = load_manifest(PROJECT / 'runs/baseline'); use = [r for r in R if r['fault_count'] in (0, 1)]
    with ProcessPoolExecutor(10) as ex: P = dict(ex.map(prof, use, chunksize=8))
    out = PROJECT / 'runs/angle_profile_20261123'; out.mkdir(parents=True, exist_ok=True); write_json(out / 'profiles.json', P)
    base = {k: np.nanmean([P[r['uid']][k] for r in use if r['fault_count'] == 0], 0) for k in BANDS}
    groups = collections.defaultdict(list); POS = ['FRONT', 'REAR', 'RIGHT']
    for r in use:
        if r['fault_count'] == 1:
            i = [c != 'Healthy' for c in r['classes']].index(True); groups[(POS[i], r['classes'][i])].append(r['uid'])
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    x = np.arange(NB) * 10 + 5; keys = sorted(groups)
    fig, ax = plt.subplots(len(BANDS), 1, figsize=(13, 3.2 * len(BANDS)), sharex=True)
    cols = {'FRONT': '#d62728', 'REAR': '#1f77b4', 'RIGHT': '#2ca02c'}; ls = {'Imbalance': '-', 'Modulation': '--', 'Whine': ':', 'PWM': '-.', 'Wind': '-'}
    summ = {}
    for j, k in enumerate(BANDS):
        for g in keys:
            d = np.nanmean([P[u][k] for u in groups[g]], 0) - base[k]
            ax[j].plot(x, d, color=cols[g[0]], ls=ls[g[1]], lw=1.6 if g[1] in ('PWM', 'Whine') else 1, label=f'{g[1]} @ {g[0]} (n={len(groups[g])})')
            summ.setdefault(k, {})[f'{g[1]}@{g[0]}'] = dict(peak_angle=int(x[np.nanargmax(d)]), peak_minus_min_db=round(float(np.nanmax(d) - np.nanmin(d)), 1))
        for a0, a1, c in [(135, 225, '#d62728'), (225, 315, '#2ca02c'), (315, 360, '#1f77b4'), (0, 45, '#1f77b4')]:
            ax[j].axvspan(a0, a1, color=c, alpha=.06)
        ax[j].set_ylabel(f'{k}\n dB vs healthy'); ax[j].axhline(0, color='k', lw=.5)
    ax[0].legend(ncol=3, fontsize=8, loc='upper left'); ax[-1].set_xlabel('microphone angle (deg)   shaded: FRONT 135-225 red, RIGHT 225-315 green, REAR 315-45 blue')
    fig.suptitle('1-fault recordings: band level vs angle, minus mean of all-healthy recordings', fontsize=13)
    plt.tight_layout(); plt.savefig(PROJECT / 'figures/angle_profile_1fault.png', dpi=85)
    write_json(out / 'summary.json', summ); print(json.dumps(summ, indent=1))
