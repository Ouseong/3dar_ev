"""Whole-recording linear STFT cache, paper-style input (Kunte et al.): 1536 freq x 384 'RPM' columns.
User request 2026-11-23. Read-only on raw data; never touches runs/baseline.

- n_fft 6144 (7.8125 Hz bins, Hann), bins 0..1535 = 0..11,992 Hz (paper: 1536 frequency rows).
- hop 480 (10 ms), centred (audio zero-padded by n_fft/2 at both ends so the edges are covered);
  the ~400 frames are linearly resampled in time to 384 columns (paper: 384 RPM rows).
- The paper's second axis is order-tracked RPM (0..10,000). We have no measured tachometer: the RPM per
  recording is the ESTIMATED source interval (median over the 3 speakers, ~5,000->7,000 or 6,000->8,000),
  assumed linear in time. Each column therefore = one time step = one estimated RPM value. Stored as rpm0/rpm1.
- Power spectrum in dBFS, floor -120 (same as baseline).
Output: runs/ae_paper_whole_20261123/cache/<uid>.npz  (spec float16 (1536,384), rpm0, rpm1, feature_key)
Usage: python build_cache_whole.py [--workers 8] [--limit N]
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
from ev3dar.features import cache_path

ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
BASE = PROJECT / 'runs/baseline'
OUT = PROJECT / 'runs/ae_paper_whole_20261123'
SPEC = dict(n_fft=6144, hop_length=480, n_freq=1536, n_cols=384, db_floor=-120.0, center=True)


def whole_key(record):
    return hashlib.sha256(json.dumps(dict(version='whole1', uid=record['uid'], cfg=SPEC,
                                          inputs=record['input_fingerprint']), sort_keys=True).encode()).hexdigest()


def extract(record, fs_expected):
    audio, fs = sf.read(data_path(ROOT, record['wav']), dtype='float32')
    if fs != fs_expected or audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError('Expected finite mono 48kHz WAV: ' + record['uid'])
    if abs(len(audio) / fs - record['duration']) > .002: raise ValueError('WAV/metadata duration mismatch.')
    n = SPEC['n_fft']
    x = np.pad(audio.astype(np.float64), (n // 2, n // 2))
    _, _, p = signal.spectrogram(x, fs=fs, window='hann', nperseg=n, noverlap=n - SPEC['hop_length'],
                                 detrend=False, scaling='spectrum', mode='psd')
    db = 10.0 * np.log10(np.maximum(p[:SPEC['n_freq']], 10.0 ** (SPEC['db_floor'] / 10.0)))
    m = db.shape[1]
    src = np.linspace(0, m - 1, SPEC['n_cols'])
    spec = np.stack([np.interp(src, np.arange(m), row) for row in db])
    rpm = np.median(np.array(record['rpm_intervals'], dtype=np.float64), axis=0)
    return dict(spec=spec.astype(np.float16), rpm0=np.float32(rpm[0]), rpm1=np.float32(rpm[1]), n_frames_raw=np.int32(m))


def one(args):
    r, fs = args
    p = cache_path(OUT, r['uid']); key = whole_key(r)
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            if str(z['feature_key'].item()) == key: return 0
    feat = extract(r, fs)
    tmp = p.with_suffix('.pending.npz'); np.savez_compressed(tmp, feature_key=np.array(key), **feat); tmp.replace(p)
    return 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument('--limit', type=int, default=None); a = ap.parse_args()
    cfg = load_config(BASE / 'config.resolved.json'); records = load_manifest(BASE)[:a.limit]
    (OUT / 'cache').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(one, [(r, cfg['sample_rate']) for r in records], chunksize=8), 1):
            if i % 200 == 0 or i == len(records): print(f'whole cache: {i}/{len(records)}', flush=True)
    write_json(OUT / 'cache/config.json', dict(SPEC, bin_hz=cfg['sample_rate'] / SPEC['n_fft']))


if __name__ == '__main__':
    main()
