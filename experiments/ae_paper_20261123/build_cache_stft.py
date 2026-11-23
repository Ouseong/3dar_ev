"""Linear-frequency STFT ROI cache for the paper AE (user request 2026-11-23). Read-only on raw data.

Same ROIs, angle->time mapping, hop (480 = 10 ms), dB floor and 100-frame cap as ev3dar.features.extract_record,
but NO mel compression: n_fft 6144 (7.8125 Hz bins, 128 ms Hann window), bins 0..1535 = 0..11,992 Hz,
i.e. 1536 frequency rows like the paper's 1536-row spectrogram. Power spectrum ('spectrum' scaling) in dBFS.
Output: runs/ae_paper_20261123/stft_cache/cache/<uid>.npz with stft_views (3,2,1536,100) float16,
id_masks, view_durations. Never touches runs/baseline.
Usage: python build_cache_stft.py [--workers 8] [--width 60]  (width != 90 -> stft_cache_roi<width>/)
"""
from pathlib import Path
import argparse, hashlib, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import soundfile as sf
from scipy import signal

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_config, load_manifest, data_path, write_json
from ev3dar.features import angle_track, circular_segments, cache_path
sys.path.insert(0, str(PROJECT / 'experiments/ae_20261119'))
from train_ae import roi_config

ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
BASE = PROJECT / 'runs/baseline'
OUT = PROJECT / 'runs/ae_paper_20261123/stft_cache'


def out_dir(width):
    return OUT if width == 90 else PROJECT / f'runs/ae_paper_20261123/stft_cache_roi{int(width)}'
STFT = dict(n_fft=6144, hop_length=480, n_freq=1536)


def stft_key(record, cfg):
    feat = dict(STFT, sample_rate=cfg['sample_rate'], db_floor=cfg['db_floor'], id_frames=cfg['id_frames'],
                positions=cfg['positions'], rois=cfg['rois'])
    return hashlib.sha256(json.dumps(dict(version='stft1', uid=record['uid'], cfg=feat,
                                          inputs=record['input_fingerprint']), sort_keys=True).encode()).hexdigest()


def log_stft(segment, fs, floor):
    n = STFT['n_fft']
    if len(segment) < n: segment = np.pad(segment, (0, n - len(segment)))
    _, _, p = signal.spectrogram(segment.astype(np.float64), fs=fs, window='hann', nperseg=n,
                                 noverlap=n - STFT['hop_length'], detrend=False, scaling='spectrum', mode='psd')
    p = p[:STFT['n_freq']]
    return (10.0 * np.log10(np.maximum(p, 10.0 ** (floor / 10.0)))).astype(np.float32)


def extract(record, cfg):
    audio, fs = sf.read(data_path(ROOT, record['wav']), dtype='float32')
    if fs != cfg['sample_rate'] or audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError('Expected finite mono 48kHz WAV: ' + record['uid'])
    duration = len(audio) / fs
    if abs(duration - record['duration']) > .002: raise ValueError('WAV/metadata duration mismatch.')
    at, aa = angle_track(data_path(ROOT, record['angles']), duration)
    P, frames, F = len(cfg['positions']), cfg['id_frames'], STFT['n_freq']
    views = np.full((P, 2, F, frames), cfg['db_floor'], np.float32)
    masks = np.zeros((P, 2, frames), np.float32); durs = np.zeros((P, 2), np.float32)
    for pi, pos in enumerate(cfg['positions']):
        for vi, (a, b) in enumerate(circular_segments(*cfg['rois'][pos])):
            t0, t1 = np.interp([a, b], aa, at)
            seg = audio[round(t0 * fs):round(t1 * fs)]
            if len(seg) < 2: raise ValueError('Empty angular ROI.')
            part = log_stft(seg, fs, cfg['db_floor']); n = part.shape[1]
            if n > frames:   # same rule as baseline: resample feature time only
                part = np.stack([np.interp(np.linspace(0, n - 1, frames), np.arange(n), v) for v in part]); n = frames
            views[pi, vi, :, :n] = part; masks[pi, vi, :n] = 1; durs[pi, vi] = t1 - t0
    return dict(stft_views=views.astype(np.float16), id_masks=masks, view_durations=durs)


def one(args):
    r, cfg, out = args
    p = cache_path(out, r['uid']); key = stft_key(r, cfg)
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            if str(z['feature_key'].item()) == key: return 0
    feat = extract(r, cfg)
    tmp = p.with_suffix('.pending.npz'); np.savez_compressed(tmp, feature_key=np.array(key), **feat); tmp.replace(p)
    return 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--limit', type=int, default=None); ap.add_argument('--width', type=float, default=90.0)
    a = ap.parse_args()
    cfg = load_config(BASE / 'config.resolved.json'); records = load_manifest(BASE)[:a.limit]
    if a.width != 90: cfg = roi_config(cfg, a.width)
    out = out_dir(a.width); (out / 'cache').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(one, [(r, cfg, out) for r in records], chunksize=8), 1):
            if i % 200 == 0 or i == len(records): print(f'stft cache: {i}/{len(records)}', flush=True)
    write_json(out / 'cache/config.json', dict(STFT, bin_hz=cfg['sample_rate'] / STFT['n_fft'], rois=cfg['rois'], width=a.width))


if __name__ == '__main__':
    main()
