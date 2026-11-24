"""Read only canonical index + long-form3-row info CSVs, never error archives."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import numpy as np
from .common import read_csv, read_json, write_json, data_path, digest, config_hash


def interval(text):
    a, b = map(float, text.split('~'))
    if not np.isfinite([a, b]).all() or b < a: raise ValueError('Invalid interval: ' + text)
    return [a, b]


def discover(root, cfg):
    root = Path(root).resolve(); index = root / '측정_목록.csv'
    records = []
    for row in read_csv(index):
        if int(row['스피커수']) != 3: continue
        paths = {key: str(row[col]).replace('\\', '/') for key, col in
                 [('wav', 'WAV'), ('info', '실험정보CSV'), ('angles', '각도CSV')]}
        for relative in paths.values():
            if not data_path(root, relative).is_file(): raise FileNotFoundError(relative)
        info = read_csv(data_path(root, paths['info']))
        if len(info) != 3 or {s['Speaker_Loc'] for s in info} != set(cfg['positions']):
            raise ValueError('Expected exactly three unique positions: ' + paths['info'])
        sources = {s['Speaker_Loc']: s for s in info}
        wav_name = Path(paths['wav']).name
        for pos, s in sources.items():
            if s['status'] != 'OK' or s['Saved_Filename'] != wav_name:
                raise ValueError('Non-successful or inconsistent capture: ' + paths['info'])
            if s['Sound_Class'] not in cfg['classes']: raise ValueError(s['Sound_Class'])
            expected = row[{'FRONT': '앞', 'REAR': '뒤', 'RIGHT': '오른쪽'}[pos]]
            if s['Sound_Class'] != expected: raise ValueError('Index/source label mismatch.')
        fault_count = sum(s['Sound_Class'] != 'Healthy' for s in info)
        if fault_count != int(row['고장음위치수']): raise ValueError('Fault count mismatch.')
        duration = float(info[0]['rotation_time_sec'])
        if not 3.7 <= duration <= 4.3: raise ValueError('Expected encoder-gated~4sec audio.')
        record = dict(uid=row['측정ID'], **paths, band=row['추정RPM구간'],
            fault_count=fault_count, source_count=3, previous_role=row['학습분할'],
            original_order=int(row['원본회차']), duration=duration,
            classes=[sources[p]['Sound_Class'] for p in cfg['positions']],
            source_files=[sources[p]['Source_File'].replace('\\', '/') for p in cfg['positions']],
            rpm_intervals=[interval(sources[p]['Estimated_RPM_Interval']) for p in cfg['positions']],
            event_time=info[0]['event_time'])
        record['input_fingerprint'] = {k: dict(bytes=data_path(root, v).stat().st_size,
            sha256=digest(data_path(root, v)))
            for k, v in paths.items()}
        records.append(record)
    records.sort(key=lambda r: r['uid'])
    if len({r['uid'] for r in records}) != len(records): raise ValueError('Duplicate recording UID.')
    return records


def source_audit(records):
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    first = {}; reuse = Counter()
    for i, r in enumerate(records):
        for source in r['source_files']:
            reuse[source] += 1
            if source in first:
                a, b = find(i), find(first[source])
                if a != b: parent[b] = a
            else: first[source] = i
    components = Counter(find(i) for i in range(len(records)))
    return dict(unique_source_files=len(reuse), reused_source_files=sum(n > 1 for n in reuse.values()),
        maximum_source_reuse=max(reuse.values(), default=0),
        connected_component_sizes=sorted(components.values(), reverse=True))


def make_split(records, seed):
    """Stratify fault-count x RPM band, BEFORE all angular/time windowing."""
    groups = defaultdict(list)
    for r in records: groups[(r['fault_count'], r['band'])].append(r['uid'])
    rng = np.random.default_rng(seed)
    split = dict(seed=seed, mode='recording_holdout_closed_source', train=[], validation=[], test=[])
    for key in sorted(groups):
        ids = np.array(sorted(groups[key]), dtype=object); rng.shuffle(ids)
        n = len(ids); a = int(.8 * n); b = int(.1 * n)
        if min(a, b, n-a-b) < 1: raise ValueError('Insufficient stratum: '+str(key))
        split['train'].extend(ids[:a].tolist()); split['validation'].extend(ids[a:a+b].tolist())
        split['test'].extend(ids[a+b:].tolist())
    for key in ['train', 'validation', 'test']: split[key].sort()
    sets = [set(split[k]) for k in ['train', 'validation', 'test']]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)): raise AssertionError('Recording leakage.')
    if set.union(*sets) != {r['uid'] for r in records}: raise AssertionError('Split coverage.')
    lookup = {r['uid']: r for r in records}
    source_sets = {k: {s for uid in split[k] for s in lookup[uid]['source_files']}
                   for k in ['train', 'validation', 'test']}
    split['source_file_overlap'] = {a+'__'+b: len(source_sets[a] & source_sets[b])
        for a, b in [('train','validation'), ('train','test'), ('validation','test')]}
    split['unseen_source_generalization_claim_allowed'] = False
    return split


def prepare(root, out, cfg):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    records = discover(root, cfg)
    groups = Counter(r['fault_count'] for r in records)
    if len(records) != 1200 or groups != {0:300, 1:300, 2:300, 3:300}:
        raise ValueError('Expected the full1200 three-speaker recordings: '+str(groups))
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    write_json(out/'config.resolved.json', cfg)
    for seed in cfg['seeds']: write_json(out/f'splits/seed{seed}.json', make_split(records, seed))
    audit = dict(recordings=len(records), by_fault_count=dict(groups),
        by_source_class=dict(Counter(c for r in records for c in r['classes'])),
        previous_roles=dict(Counter(r['previous_role'] for r in records)), **source_audit(records),
        split_sizes={k:len(make_split(records, cfg['seeds'][0])[k]) for k in ['train','validation','test']},
        config_hash=config_hash(cfg), geometry_status=cfg['geometry_status'],
        warnings=['All recordings have3sources at the same layout: count/DOA labels are constant.',
                  'Source audio is reused across recordings; this is CLOSED-SOURCE recording holdout.',
                  'RPM values are estimated metadata, not tachometer ground truth.',
                  'A sound-type label does not identify a physically damaged component.'])
    write_json(out/'data_audit.json', audit)
    return audit
