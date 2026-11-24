from pathlib import Path
import csv
import hashlib
import json

PROJECT = Path(__file__).resolve().parents[1]


def load_config(path=None):
    path = Path(path) if path else PROJECT / 'configs/ev.json'
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    if cfg['angular_bins'] != 360 or cfg['sample_rate'] != 48000:
        raise ValueError('This baseline expects360 angular bins and48kHz recordings.')
    if cfg['classes'][0] != 'Healthy' or len(cfg['classes']) != 6:
        raise ValueError('Expected Healthy followed by the five dataset sound types.')
    if cfg['split_mode'] != 'recording_holdout_closed_source':
        raise ValueError('Strict source-disjoint8:1:1 cannot use all1200 recordings; see audit.')
    return cfg


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1048576), b''): h.update(part)
    return h.hexdigest()


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def data_path(root, relative):
    """Normalize Windows CSV separators on Linux; never escape the dataset root."""
    root = Path(root).resolve()
    relative = str(relative).replace('\\', '/')
    if relative.startswith('/') or ':' in relative or '..' in Path(relative).parts:
        raise ValueError('Expected a portable relative dataset path: ' + relative)
    path = (root / relative).resolve()
    if not path.is_relative_to(root): raise ValueError('Dataset path escaped root.')
    return path


def load_manifest(out):
    path = Path(out) / 'manifest.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def records_by_ids(records, ids):
    lookup = {r['uid']: r for r in records}
    return [lookup[uid] for uid in ids]
