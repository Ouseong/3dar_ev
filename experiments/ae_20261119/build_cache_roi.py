"""Rebuild the per-ROI spectrogram cache for a narrower ROI width (read-only on raw data).

Same feature code (ev3dar.features.extract_record) and same ROI centres as the baseline;
only cfg['rois'] changes. Output: <out>/cache/<uid>.npz, never touches runs/baseline.
Usage: python build_cache_roi.py --width 60 --out runs/ae_roi60_20261119
"""
from pathlib import Path
import argparse, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from ev3dar.common import load_config, load_manifest, write_json, config_hash
from ev3dar.features import extract_record, feature_key, cache_path
from train_ae import roi_config, BASE

ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')


def one(args):
    r, cfg, out = args
    p = cache_path(out, r['uid']); key = feature_key(r, cfg)
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            if str(z['feature_key'].item()) == key: return 0
    feat = extract_record(ROOT, r, cfg)
    tmp = p.with_suffix('.pending.npz'); np.savez_compressed(tmp, feature_key=np.array(key), **feat); tmp.replace(p)
    return 1


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--width', type=float, required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    cfg = roi_config(load_config(BASE / 'config.resolved.json'), a.width); records = load_manifest(BASE)
    out = Path(a.out); (out / 'cache').mkdir(parents=True, exist_ok=True)
    done = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for i, n in enumerate(ex.map(one, [(r, cfg, out) for r in records], chunksize=8), 1):
            done += n
            if i % 200 == 0 or i == len(records): print(f'width {a.width}: {i}/{len(records)}', flush=True)
    write_json(out / 'cache/config.json', dict(config_hash=config_hash(cfg), rois=cfg['rois'], width=a.width))


if __name__ == '__main__':
    main()
