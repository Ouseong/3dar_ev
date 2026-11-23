"""Whole-recording 1536 x 384 linear-STFT cache for the 200 1-speaker recordings.
Same extraction as experiments/ae_paper_whole_20261123/build_cache_whole.py (n_fft 6144, hop 480, centred,
resampled to 384 columns, dBFS floor -120). Output: runs/ae_paper_single_20261123/cache/<uid>.npz
"""
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT / 'experiments/ae_paper_whole_20261123'))
from ev3dar.common import read_json
from ev3dar.features import cache_path
from build_cache_whole import extract, whole_key

OUT = PROJECT / 'runs/ae_paper_single_20261123'


def one(r):
    p = cache_path(OUT, r['uid']); key = whole_key(r)
    if p.exists():
        with np.load(p, allow_pickle=False) as z:
            if str(z['feature_key'].item()) == key: return 0
    feat = extract(r, 48000)
    tmp = p.with_suffix('.pending.npz'); np.savez_compressed(tmp, feature_key=np.array(key), **feat); tmp.replace(p)
    return 1


if __name__ == '__main__':
    recs = read_json(OUT / 'manifest.json'); (OUT / 'cache').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(8) as ex: n = sum(ex.map(one, recs, chunksize=4))
    print('built', n, 'of', len(recs))
