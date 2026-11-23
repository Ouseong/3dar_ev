"""Direct pixel comparison, 1-speaker Healthy vs Fault (no model). User request 2026-11-23.
Input: whole-recording 1536 x 384 dB spectrograms (build_cache_single.py), clipped to -120..0 dBFS.
Comparisons are made within the same speaker position and RPM band (10 Healthy recordings per group),
so that time columns line up (same rotation start, same estimated RPM range).
  - healthy reference = mean of the group's Healthy images; a Healthy recording is compared leave-one-out.
  - pixel RMS difference (dB) to the reference; same after time averaging (1536-bin mean spectrum).
  - fraction of pixels outside the Healthy range (|x - mean| > 3 x pixel std of Healthy, leave-one-out).
Outputs: figures/single_pixel_compare.png, runs/ae_paper_single_20261123/pixel_compare.json
--feature mel: same comparison on 64-mel log spectrograms (baseline config: n_fft 2048, hop 480, 0-12 kHz, centred,
  resampled to 384 columns) -> figures/single_pixel_compare_mel.png, pixel_compare_mel.json
--feature stft6k: linear STFT n_fft 12288 (3.906 Hz bins, 256 ms Hann), 0-6 kHz = 1536 rows like the paper figures,
  hop 480 centred, resampled to 384 columns -> figures/single_pixel_compare_stft6k.png, pixel_compare_stft6k.json
"""
from pathlib import Path
import sys, json
from collections import defaultdict
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ev3dar.common import read_json, write_json
from ev3dar.features import cache_path

OUT = PROJECT / 'runs/ae_paper_single_20261123'
BIN_HZ = 48000 / 6144
FEAT = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == '--feature' else 'stft'
MEL = FEAT == 'mel'
recs = read_json(OUT / 'manifest.json')
if MEL:
    import soundfile as sf
    from ev3dar.common import load_config, data_path
    from ev3dar.features import logmel, mel_bank
    cfg = load_config(PROJECT / 'runs/baseline/config.resolved.json')
    ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
    def mel_img(r):
        a, fs = sf.read(data_path(ROOT, r['wav']), dtype='float32'); n = cfg['n_fft']
        db, _ = logmel(np.pad(a, (n // 2, n // 2)), cfg); src = np.linspace(0, db.shape[1] - 1, 384)
        return np.clip(np.stack([np.interp(src, np.arange(db.shape[1]), row) for row in db]), -120, 0).astype(np.float32)
    X = {r['uid']: mel_img(r) for r in recs}
    w = mel_bank(cfg); FREQS = (w * np.fft.rfftfreq(cfg['n_fft'], 1 / 48000)[None]).sum(1) / w.sum(1)   # mel band centre (Hz)
elif FEAT == 'stft6k':
    import soundfile as sf
    from scipy import signal
    from ev3dar.common import data_path
    ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
    def stft6k(r):
        a, fs = sf.read(data_path(ROOT, r['wav']), dtype='float32'); n = 12288
        _, _, p = signal.spectrogram(np.pad(a.astype(np.float64), (n // 2, n // 2)), fs=fs, window='hann', nperseg=n,
                                     noverlap=n - 480, detrend=False, scaling='spectrum', mode='psd')
        db = 10 * np.log10(np.maximum(p[:1536], 1e-12)); src = np.linspace(0, db.shape[1] - 1, 384)
        return np.clip(np.stack([np.interp(src, np.arange(db.shape[1]), row) for row in db]), -120, 0).astype(np.float32)
    X = {r['uid']: stft6k(r) for r in recs}
    FREQS = np.arange(1536) * 48000 / 12288
else:
    X = {r['uid']: np.clip(np.load(cache_path(OUT, r['uid']))['spec'].astype(np.float32), -120, 0) for r in recs}
    FREQS = np.arange(1536) * BIN_HZ
TAG = '' if FEAT == 'stft' else '_' + FEAT
groups = defaultdict(lambda: defaultdict(list))
for r in recs: groups[(r['position'], r['band'])][r['sound_class']].append(r['uid'])

rows = defaultdict(lambda: defaultdict(list)); diff_img = defaultdict(list); diff_spec = defaultdict(list); hh_spec = []
for (pos, band), g in groups.items():
    H = np.stack([X[u] for u in g['Healthy']])
    for i in range(len(H)):                                    # Healthy, leave-one-out
        ref = np.delete(H, i, 0); m, s = ref.mean(0), ref.std(0) + 1e-3
        d = H[i] - m
        rows['Healthy']['rms_pixel_db'].append(float(np.sqrt((d ** 2).mean())))
        rows['Healthy']['rms_spectrum_db'].append(float(np.sqrt((d.mean(1) ** 2).mean())))
        rows['Healthy']['frac_out_3sd'].append(float((np.abs(d) > 3 * s).mean()))
        hh_spec.append(d.mean(1))
    m, s = H.mean(0), H.std(0) + 1e-3
    for c, uids in g.items():
        if c == 'Healthy': continue
        for u in uids:
            d = X[u] - m
            rows[c]['rms_pixel_db'].append(float(np.sqrt((d ** 2).mean())))
            rows[c]['rms_spectrum_db'].append(float(np.sqrt((d.mean(1) ** 2).mean())))
            rows[c]['frac_out_3sd'].append(float((np.abs(d) > 3 * s).mean()))
            diff_img[c].append(d); diff_spec[c].append(d.mean(1))

summary = {}
for c, v in rows.items():
    summary[c] = {k: dict(mean=float(np.mean(a)), min=float(np.min(a)), max=float(np.max(a))) for k, a in v.items()}
    summary[c]['n'] = len(v['rms_pixel_db'])
    if c != 'Healthy':
        ms = np.mean(diff_spec[c], 0); top = np.argsort(-np.abs(ms))[:5]
        summary[c]['top_mean_spectrum_diff'] = [dict(hz=round(float(FREQS[i]), 1), db=round(float(ms[i]), 2)) for i in top]
hh_sd = np.std(hh_spec, 0)
write_json(OUT / f'pixel_compare{TAG}.json', summary)
for c in ['Healthy', 'Imbalance', 'Modulation', 'Whine', 'PWM', 'Wind']:
    s = summary[c]
    print(f"{c:10s} n={s['n']:3d}  pixel RMS {s['rms_pixel_db']['mean']:5.2f} dB  spectrum RMS {s['rms_spectrum_db']['mean']:5.2f} dB  "
          f"pixels >3sd {100 * s['frac_out_3sd']['mean']:5.2f}%  " + ('' if c == 'Healthy' else 'top: ' + ', '.join(f"{t['hz']:.0f}Hz {t['db']:+.1f}" for t in s['top_mean_spectrum_diff'][:3])))

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
cls = ['Imbalance', 'Modulation', 'Whine', 'PWM', 'Wind']
fig, ax = plt.subplots(len(cls), 2, figsize=(15, 3.0 * len(cls)), gridspec_kw=dict(width_ratios=[1, 1.4]))
f = FREQS
for i, c in enumerate(cls):
    im = ax[i, 0].imshow(np.mean(diff_img[c], 0), origin='lower', aspect='auto', cmap='RdBu_r', vmin=-15, vmax=15,
                         extent=[0, 384, 0, len(f)] if MEL else [0, 384, 0, f[-1] / 1000])
    ax[i, 0].set_title(f'{c}: mean(fault) − mean(healthy), dB' + {'mel': ' (64 mel)', 'stft': ' (STFT 0-12 kHz)', 'stft6k': ' (STFT 0-6 kHz)'}[FEAT]); ax[i, 0].set_ylabel('mel bin' if MEL else 'kHz'); plt.colorbar(im, ax=ax[i, 0])
    ms = np.mean(diff_spec[c], 0)
    ax[i, 1].fill_between(f / 1000, -2 * hh_sd, 2 * hh_sd, color='0.8', label='healthy vs healthy ±2σ')
    ax[i, 1].plot(f / 1000, ms, lw=0.8, color='C3', label=f'{c} − healthy (time-averaged)')
    ax[i, 1].axhline(0, color='k', lw=0.5); ax[i, 1].set_ylim(-20, 25); ax[i, 1].set_xlim(0, FREQS[-1] / 1000)
    ax[i, 1].set_ylabel('dB'); ax[i, 1].legend(loc='upper right', fontsize=8)
ax[-1, 0].set_xlabel('time column (≈ estimated RPM, 384)'); ax[-1, 1].set_xlabel('frequency (kHz)')
plt.tight_layout(); plt.savefig(PROJECT / f'figures/single_pixel_compare{TAG}.png', dpi=100)
