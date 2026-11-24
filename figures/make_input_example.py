"""Input example for the 3x3 FaultBandNet: one test recording (default 1102_0324: rear = PWM).
(1) STFT of the whole recording with the three bands marked, mic angle on the top axis
(2) the three maps actually fed to the network (from the feature cache) + the 3x3 output (notebook test probabilities)
Also writes single-map PNGs used by make_faultbandnet_model_io.py.  Usage: python make_input_example.py [uid]
"""
import sys, json, numpy as np, soundfile as sf
from pathlib import Path
from scipy import signal
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
PROJECT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT / 'experiments/knowledge_rules_20261123'))
from ev3dar.common import load_manifest, data_path
from ev3dar.features import angle_track
import knowledge_detector as K
UID = sys.argv[1] if len(sys.argv) > 1 else '1102_0324'
ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV'); OUT = PROJECT / 'figures/input_example'
NB = PROJECT / 'runs/nb_faultbandnet_all_20261124'
R = load_manifest(PROJECT / 'runs/baseline'); r = next(x for x in R if x['uid'] == UID)
z = np.load(PROJECT / 'runs/nb_faultbandnet_20261124/features_81b146e716ad05ea.npz'); k = list(z['uid']).index(UID)
maps = [z['wind'][k], z['pwm'][k], z['whine'][k]]
keep = [x['uid'] for x in R]                                   # notebook uses all 1,200 recordings (sorted by uid)
P = np.load(NB / 'test_probabilities_3x3.npy')[keep.index(UID)]
ADJ = {q['uid']: q['rpm_adj'] for q in json.load(open(PROJECT / 'runs/knowledge_rules_20261123/roi_features_refined.json'))}

a, fs = sf.read(data_path(ROOT, r['wav']), dtype='float32'); dur = len(a) / fs; at, aa = angle_track(data_path(ROOT, r['angles']), dur)
f, t, S = signal.spectrogram(a.astype(np.float64), fs=fs, window='hann', nperseg=8192, noverlap=8192 - 480, mode='psd')
db = 10 * np.log10(S + 1e-30); ang = np.interp(t, at, aa)
r0, r1 = np.median(np.array(r['rpm_intervals']), axis=0); q = ADJ[UID]; rpm = K.make_rpm(r0 * (1 + q[0]), r1 * (1 + q[1]), q[2], dur)(t)

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 9})
fig = plt.figure(figsize=(11, 7.2)); gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], width_ratios=[1, 1, 1, 0.8], hspace=0.55, wspace=0.45)
ax = fig.add_subplot(gs[0, :]); fm = f <= 8000
ax.pcolormesh(t, f[fm] / 1000, db[fm], cmap='magma', vmin=np.percentile(db[fm], 30), vmax=np.percentile(db[fm], 99.7), shading='auto')
ax.axhline(3.3, color='#9ad9c5', lw=1.4); ax.axhline(6.2, color='#9ad9c5', lw=1.4)
ax.axhline(4.8, color='#f4a259', lw=1, ls='--'); ax.axhline(5.2, color='#f4a259', lw=1, ls='--')
ax.plot(t, (16 * rpm / 60 - 150) / 1000, color='#8ecae6', lw=1); ax.plot(t, (16 * rpm / 60 + 150) / 1000, color='#8ecae6', lw=1); 
ax.text(t[-1] * 1.005, 6.0, 'Wind\n3.3–6.2 kHz', color='#2a8a6e', fontsize=8, va='center'); ax.text(t[-1] * 1.005, 5.0, 'PWM\n4.8–5.2 kHz', color='#c0661b', fontsize=8, va='center')
ax.text(t[-1] * 1.005, 16 * rpm[-1] / 60 / 1000, 'Whine\norder 16 ±150 Hz', color='#2b7aa1', fontsize=8, va='center')
ax.set_xlabel('Time (s)'); ax.set_ylabel('Frequency (kHz)'); ax.set_xlim(0, t[-1])
top = ax.twiny(); top.set_xlim(0, t[-1]); tk = [np.interp(d, aa, at) for d in (0, 90, 180, 270, 360)]; top.set_xticks(tk); top.set_xticklabels(['0°', '90°', '180°', '270°', '360°']); top.set_xlabel('Mic angle (one rotation)')
ax.set_title(f'(1) STFT of the whole recording {UID}  —  front {r["classes"][0]}, rear {r["classes"][1]}, right {r["classes"][2]};  RPM {rpm[0]:.0f}→{rpm[-1]:.0f}', fontsize=9, pad=28)
ext = [(3.3, 6.2), (4.8, 5.2), (-150, 150)]; nm = ['Wind map 36×58', 'PWM map 36×68', 'Whine map 36×51']; xl = ['kHz', 'kHz', 'Hz from order-16 line']
for j in range(3):
    ax = fig.add_subplot(gs[1, j]); im = ax.imshow(maps[j], origin='lower', aspect='auto', cmap='magma', extent=[*ext[j], 0, 360], vmin=-3, vmax=12)
    ax.set_yticks([0, 90, 180, 270, 360]); ax.set_xlabel(xl[j]); ax.set_title(nm[j], fontsize=9)
    if j == 0: ax.set_ylabel('Mic angle (°)')
cax = ax.inset_axes([1.04, 0, 0.05, 1]); cb = fig.colorbar(im, cax=cax); cb.set_label('dB (relative)', fontsize=7)
ax = fig.add_subplot(gs[1, 3]); ax.imshow(P, cmap='Reds', vmin=0, vmax=1)
ax.set_xticks(range(3)); ax.set_xticklabels(['Whine', 'PWM', 'Wind'], fontsize=8); ax.set_yticks(range(3)); ax.set_yticklabels([f'{p}\n({c})' for p, c in zip(['Front', 'Rear', 'Right'], r['classes'])], fontsize=7)
for i in range(3):
    for j in range(3): ax.text(j, i, f'{P[i, j]:.2f}', ha='center', va='center', fontsize=8, color='white' if P[i, j] > .6 else 'black')
ax.yaxis.tick_right(); ax.set_title('Output 3×3 (0–1 each)', fontsize=9)
fig.text(0.01, 0.43, '(2) Network input: each 10° angle bin of (1)\n     → one row of each map', fontsize=9)
OUT.mkdir(exist_ok=True); fig.savefig(OUT / f'input_example_{UID}.png', dpi=160, bbox_inches='tight')
for j, n in enumerate(['wind', 'pwm', 'whine']):
    g = plt.figure(figsize=(1.6, 1.6)); g.add_axes([0, 0, 1, 1]).imshow(maps[j], origin='lower', aspect='auto', cmap='magma', vmin=-3, vmax=12); plt.axis('off')
    g.savefig(OUT / f'map_{n}.png', dpi=90); plt.close(g)
json.dump(dict(uid=UID, classes=r['classes'][:3], probs=np.round(P, 2).tolist()), open(OUT / 'example.json', 'w'))
print('ok', P.round(2).tolist())
