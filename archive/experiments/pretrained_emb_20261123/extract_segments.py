"""Cut every ROI view of the 1200 3-speaker recordings and resample to 16 kHz (input of BEATs / YAMNet).
User request 2026-11-23: try the penguin notebook's pretrained extractors (BEATs, YAMNet) as a SEPARATE experiment
(CLAUDE.md item 4 excludes them from the baseline; existing code untouched). Read-only on raw data.
ROIs = baseline (FRONT 135-225, RIGHT 225-315, REAR 315-360 | 0-45 as two separate views, never spliced).
Resampling 48 k -> 16 k with scipy resample_poly (anti-alias at 8 kHz: content above 8 kHz is lost).
Output: runs/pretrained_emb_20261123/segments_16k.npz  (uid, position, view, duration, audio as object array)
"""
from pathlib import Path
import sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_config, load_manifest, data_path
from ev3dar.features import angle_track, circular_segments
ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
cfg = load_config(PROJECT / 'runs/baseline/config.resolved.json')
out = PROJECT / 'runs/pretrained_emb_20261123'; out.mkdir(parents=True, exist_ok=True)
U, PO, V, D, A = [], [], [], [], []
for r in load_manifest(PROJECT / 'runs/baseline'):
    a, fs = sf.read(data_path(ROOT, r['wav']), dtype='float32'); at, aa = angle_track(data_path(ROOT, r['angles']), len(a) / fs)
    for pos in cfg['positions']:
        for vi, (lo, hi) in enumerate(circular_segments(*cfg['rois'][pos])):
            t0, t1 = np.interp([lo, hi], aa, at); seg = a[round(t0 * fs):round(t1 * fs)]
            U.append(r['uid']); PO.append(pos); V.append(vi); D.append(t1 - t0); A.append(resample_poly(seg, 1, 3).astype(np.float32))
arr = np.empty(len(A), dtype=object); arr[:] = A
np.savez(out / 'segments_16k.npz', uid=np.array(U), position=np.array(PO), view=np.array(V), duration=np.array(D), audio=arr)
print('views', len(A), 'median length s', np.median([len(x) for x in A]) / 16000)
