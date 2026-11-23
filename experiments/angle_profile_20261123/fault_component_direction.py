"""Does each fault's OWN component get louder when the mic faces the fault speaker? (user request 2026-11-23)
Per recording, per STFT frame (n_fft 8192 = 5.9 Hz bins, hop 480), component power:
  whine_o16 : order 16 +-15 Hz (RPM = estimated interval, linear in time)    pwm_5k : 4990-5010 Hz
  imb_o1o2  : orders 1 and 2 +-8 Hz                                          mod_o8 : order 8 +-20 Hz
  wind_band : 3750-5750 Hz                                                   common_0.7-2.5k : 700-2500 Hz
binned into 36 x 10-deg mic-angle bins (dB). For each fault class, shape(position p) = mean profile of class-at-p
minus mean over the positions available for that class (cancels the run-up time envelope; needs >= 2 positions).
Wind exists only at RIGHT: its shape is taken against the mean Healthy profile of the 3 positions (same set).
facing-minus-away = mean shape within +-45 deg of the speaker minus mean beyond 135 deg (assumed FRONT 180, REAR 0, RIGHT 270).
Sets: 1-speaker recordings (clean) and 3-speaker 1-fault recordings.
Output: runs/angle_profile_20261123/fault_component_direction.json, figures/fault_component_direction.png
"""
from pathlib import Path
import sys, json, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf
from scipy import signal
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_manifest, read_json, data_path, write_json
from ev3dar.features import angle_track
ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV'); NB = 36; NF = 8192
COMP = ['whine_o16', 'pwm_5k', 'imb_o1o2', 'mod_o8', 'wind_band', 'common_0.7-2.5k']
CENTRE = {'FRONT': 180, 'REAR': 0, 'RIGHT': 270}
OWN = {'Whine': 'whine_o16', 'PWM': 'pwm_5k', 'Imbalance': 'imb_o1o2', 'Modulation': 'mod_o8', 'Wind': 'wind_band'}


def prof(args):
    r, pos, cls = args
    a, fs = sf.read(data_path(ROOT, r['wav']), dtype='float32'); dur = len(a) / fs; at, aa = angle_track(data_path(ROOT, r['angles']), dur)
    f, t, p = signal.spectrogram(a.astype(np.float64), fs=fs, window='hann', nperseg=NF, noverlap=NF - 480, scaling='spectrum', mode='psd')
    r0, r1 = np.median(np.array(r['rpm_intervals']), axis=0); hz = (r0 + (r1 - r0) * t / dur) / 60
    def track(order, half): return np.array([p[np.abs(f - order * hz[i]) <= half, i].sum() for i in range(len(t))])
    band = lambda lo, hi: p[(f >= lo) & (f <= hi)].sum(0)
    C = dict(whine_o16=track(16, 15), pwm_5k=band(4990, 5010), imb_o1o2=track(1, 8) + track(2, 8), mod_o8=track(8, 20),
             wind_band=band(3750, 5750), **{'common_0.7-2.5k': band(700, 2500)})
    b = np.clip((np.interp(t, at, aa) // 10).astype(int), 0, NB - 1)
    return r['uid'], pos, cls, {k: [float(10 * np.log10(v[b == i].mean() + 1e-20)) for i in range(NB)] for k, v in C.items()}


def shapes(P, label):
    """P: list of (pos, cls, profile). Returns {cls: {pos: {comp: shape}}} using the rule in the docstring."""
    g = collections.defaultdict(list)
    for pos, cls, pr in P: g[(cls, pos)].append(pr)
    mean = {k: {c: np.mean([x[c] for x in v], 0) for c in COMP} for k, v in g.items()}
    out = collections.defaultdict(dict)
    for cls in sorted({k[0] for k in mean}):
        poss = [p for p in CENTRE if (cls, p) in mean]
        refs = poss if len(poss) >= 2 else [p for p in CENTRE if ('Healthy', p) in mean]
        refcls = cls if len(poss) >= 2 else 'Healthy'
        if not refs: continue
        for p in poss:
            out[cls][p] = {c: mean[(cls, p)][c] - np.mean([mean[(refcls, q)][c] for q in refs], 0) for c in COMP}
    return out


def facing(sh, pos):
    x = np.arange(NB) * 10 + 5; d = np.abs(((x - CENTRE[pos]) + 180) % 360 - 180)
    return float(sh[d <= 45].mean() - sh[d >= 135].mean()), int(x[np.argmax(sh)])


if __name__ == '__main__':
    R1 = read_json(PROJECT / 'runs/ae_paper_single_20261123/manifest.json')
    R3 = [r for r in load_manifest(PROJECT / 'runs/baseline') if r['fault_count'] in (0, 1)]
    jobs = [(r, r['position'], r['sound_class'], '1spk') for r in R1]
    for r in R3:
        if r['fault_count'] == 0: jobs += [(r, p, 'Healthy', '3spk') for p in CENTRE][:1]
        else:
            i = [c != 'Healthy' for c in r['classes']].index(True); jobs.append((r, ['FRONT', 'REAR', 'RIGHT'][i], r['classes'][i], '3spk'))
    with ProcessPoolExecutor(10) as ex: res = list(ex.map(prof, [j[:3] for j in jobs], chunksize=4))
    sets = {'1spk': [], '3spk': []}
    for (r, pos, cls, s), (_, _, _, pr) in zip(jobs, res): sets[s].append((pos, cls, pr))
    # 3-speaker 0-fault recordings have all 3 positions healthy: use them as Healthy at every position
    sets['3spk'] += [(p, 'Healthy', pr) for (pos, cls, pr) in list(sets['3spk']) if cls == 'Healthy' for p in ('REAR', 'RIGHT')]
    summ = {}
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 5, figsize=(24, 8), sharex=True); x = np.arange(NB) * 10 + 5
    cols = {'FRONT': '#d62728', 'REAR': '#1f77b4', 'RIGHT': '#2ca02c'}
    for row, s in enumerate(['1spk', '3spk']):
        sh = shapes(sets[s], s)
        for j, (cls, comp) in enumerate(OWN.items()):
            a = ax[row, j]
            for pos, d in sh.get(cls, {}).items():
                fm, pk = facing(d[comp], pos); fc_common, _ = facing(d['common_0.7-2.5k'], pos)
                summ.setdefault(s, {}).setdefault(cls, {})[pos] = dict(component=comp, facing_minus_away_db=round(fm, 1), peak_angle=pk,
                                                                      common_band_facing_minus_away_db=round(fc_common, 1))
                a.plot(x, d[comp], color=cols[pos], lw=1.8, label=f'{cls} @ {pos}: facing−away {fm:+.1f} dB')
                a.axvline(CENTRE[pos], color=cols[pos], ls=':', lw=1)
            a.axhline(0, color='k', lw=.5); a.set_title(f'{"1-speaker" if s == "1spk" else "3-speaker 1-fault"}: {cls} — {comp}', fontsize=10)
            a.legend(fontsize=8); a.set_ylabel('dB (time envelope removed)')
    for a in ax[1]: a.set_xlabel('mic angle (deg)   dotted = assumed speaker direction')
    plt.tight_layout(); plt.savefig(PROJECT / 'figures/fault_component_direction.png', dpi=80)
    write_json(PROJECT / 'runs/angle_profile_20261123/fault_component_direction.json', summ); print(json.dumps(summ, indent=1))
