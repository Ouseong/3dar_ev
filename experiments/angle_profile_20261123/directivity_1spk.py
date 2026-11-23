"""Microphone directivity measured from 1-speaker recordings (user request 2026-11-23).
Band level vs angle (36 x 10 deg, same as angle_profile.py) for the 200 1-speaker recordings.
Directional contrast for position p = mean profile of p-only Healthy recordings minus mean over the 3 positions
(the run-up time envelope is common to all positions and cancels). Also for fault classes of 1-speaker set.
Output: figures/directivity_1speaker.png, runs/angle_profile_20261123/directivity_1spk.json
"""
from pathlib import Path
import sys, json, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(Path(__file__).parent))
from ev3dar.common import read_json, write_json
import angle_profile as A
A.BANDS = {'0.1-0.3k': (100, 300), '0.3-0.7k': (300, 700), '0.7-1.5k': (700, 1500), '1.5-2.5k': (1500, 2500),
           '2.5-3.75k': (2500, 3750), '3.75-5.75k': (3750, 5750), '5.75-8k': (5750, 8000), '8-12k': (8000, 12000)}
if __name__ == '__main__':
    R = read_json(PROJECT / 'runs/ae_paper_single_20261123/manifest.json')
    with ProcessPoolExecutor(10) as ex: P = dict(ex.map(A.prof, R, chunksize=4))
    write_json(PROJECT / 'runs/angle_profile_20261123/profiles_1spk.json', P)
    x = np.arange(36) * 10 + 5; POS = ['FRONT', 'REAR', 'RIGHT']; centre = {'FRONT': 180, 'REAR': 0, 'RIGHT': 270}
    H = {p: {k: np.nanmean([P[r['uid']][k] for r in R if r['position'] == p and r['sound_class'] == 'Healthy'], 0) for k in A.BANDS} for p in POS}
    summ = {}
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(20, 8), sharex=True); cols = {'FRONT': '#d62728', 'REAR': '#1f77b4', 'RIGHT': '#2ca02c'}
    for j, k in enumerate(A.BANDS):
        common = np.mean([H[p][k] for p in POS], 0); a = ax.flat[j]
        for p in POS:
            d = H[p][k] - common; a.plot(x, d, color=cols[p], lw=1.8, label=p + ' only')
            c = centre[p]; dist = np.abs(((x - c) + 180) % 360 - 180)
            summ.setdefault(k, {})[p] = dict(facing_minus_away_db=round(float(d[dist <= 45].mean() - d[dist >= 135].mean()), 2),
                                             peak_angle=int(x[np.argmax(d)]))
        for a0, a1, c in [(135, 225, '#d62728'), (225, 315, '#2ca02c'), (315, 360, '#1f77b4'), (0, 45, '#1f77b4')]: a.axvspan(a0, a1, color=c, alpha=.07)
        a.axhline(0, color='k', lw=.5); a.set_title(k + ' Hz'); a.set_ylabel('dB vs mean of 3 positions')
    ax.flat[0].legend(); fig.suptitle('Mic directivity from 1-speaker HEALTHY recordings (time envelope cancelled)', fontsize=14)
    for a in ax[1]: a.set_xlabel('mic angle (deg)')
    plt.tight_layout(); plt.savefig(PROJECT / 'figures/directivity_1speaker.png', dpi=80)
    write_json(PROJECT / 'runs/angle_profile_20261123/directivity_1spk.json', summ); print(json.dumps(summ, indent=1))
