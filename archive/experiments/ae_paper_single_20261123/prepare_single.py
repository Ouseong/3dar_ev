"""1-speaker recordings (200) for the paper-style whole-recording AE (user request 2026-11-23).
CLAUDE.md item 3 excludes the 1-speaker set from the supervised baseline; the user explicitly asked to use it here
as a separate experiment. Read-only on raw data.

Manifest: same checks as ev3dar.data.discover but exactly one playing speaker (info CSV has 1 row).
Splits (seed 42/43/44), recording level:
  Healthy 60 = 6 strata (position x RPM band) x 10 -> per stratum 6 train / 2 validation / 2 test  => 36 / 12 / 12
  Fault 140  = 14 strata (class x position x band) x 10 -> per stratum 5 validation / 5 test      => 70 / 70
  Faults are never used for training. Validation faults are used only for the 'real-label' selection comparison.
Output: runs/ae_paper_single_20261123/{manifest.json, splits/seed*.json}
"""
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ev3dar.common import read_csv, write_json, data_path, digest
from ev3dar.data import interval

ROOT = Path('/home/kos/Desktop/3DAR_EV/[3DAR Electric Car]/3DAR_EV')
OUT = PROJECT / 'runs/ae_paper_single_20261123'
POS = {'앞': 'FRONT', '뒤': 'REAR', '오른쪽': 'RIGHT'}


def discover():
    recs = []
    for row in read_csv(ROOT / '측정_목록.csv'):
        if int(row['스피커수']) != 1: continue
        paths = {k: str(row[c]).replace('\\', '/') for k, c in [('wav', 'WAV'), ('info', '실험정보CSV'), ('angles', '각도CSV')]}
        for v in paths.values():
            if not data_path(ROOT, v).is_file(): raise FileNotFoundError(v)
        info = read_csv(data_path(ROOT, paths['info']))
        if len(info) != 1: raise ValueError('Expected one playing speaker: ' + paths['info'])
        s = info[0]
        playing = [(POS[k], row[k]) for k in POS if row[k] != '재생안함']
        if len(playing) != 1 or playing[0] != (s['Speaker_Loc'], s['Sound_Class']): raise ValueError('Index/info mismatch ' + row['측정ID'])
        if s['status'] != 'OK' or s['Saved_Filename'] != Path(paths['wav']).name: raise ValueError('Bad capture ' + row['측정ID'])
        dur = float(s['rotation_time_sec'])
        if not 3.7 <= dur <= 4.3: raise ValueError('Expected ~4 s audio.')
        r = dict(uid=row['측정ID'], **paths, band=row['추정RPM구간'], position=s['Speaker_Loc'], sound_class=s['Sound_Class'],
                 fault_count=int(s['Sound_Class'] != 'Healthy'), classes=[s['Sound_Class']], source_count=1,
                 previous_role=row['학습분할'], duration=dur, source_files=[s['Source_File'].replace('\\', '/')],
                 rpm_intervals=[interval(s['Estimated_RPM_Interval'])], event_time=s['event_time'])
        r['input_fingerprint'] = {k: dict(bytes=data_path(ROOT, v).stat().st_size, sha256=digest(data_path(ROOT, v))) for k, v in paths.items()}
        recs.append(r)
    recs.sort(key=lambda r: r['uid'])
    if len(recs) != 200 or len({r['uid'] for r in recs}) != 200: raise ValueError('Expected 200 unique 1-speaker recordings.')
    return recs


def split(recs, seed):
    rng = np.random.default_rng(seed); groups = defaultdict(list)
    for r in recs: groups[(r['sound_class'], r['position'], r['band'])].append(r['uid'])
    out = dict(train=[], validation=[], test=[])
    for key in sorted(groups):
        ids = sorted(groups[key]); rng.shuffle(ids)
        if key[0] == 'Healthy':
            n_tr, n_va = round(len(ids) * .6), round(len(ids) * .2)
            out['train'] += ids[:n_tr]; out['validation'] += ids[n_tr:n_tr + n_va]; out['test'] += ids[n_tr + n_va:]
        else:
            h = len(ids) // 2; out['validation'] += ids[:h]; out['test'] += ids[h:]
    return {k: sorted(v) for k, v in out.items()}


def main():
    recs = discover(); (OUT / 'splits').mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'manifest.json', recs)
    for seed in (42, 43, 44): write_json(OUT / f'splits/seed{seed}.json', dict(seed=seed, **split(recs, seed)))
    src = defaultdict(int)
    for r in recs: src[r['source_files'][0]] += 1
    print('recordings', len(recs), 'unique source files', len(src), 'max reuse', max(src.values()))


if __name__ == '__main__':
    main()
