"""Experiment 2: keep the existing ID architecture/training loop, change ONLY input normalization.

Unchanged: ev3dar.models.SoundTypeCNN (CNN -> MLP/64-d latent -> 6-class decoder), optimizer, lr, batch,
epochs/patience, validation-macroF1 model selection, seed splits, feature cache (runs/baseline/cache, read-only).
ev3dar package files are imported, never edited. Outputs: runs/exp_fullrot_20261116/<variant>/seed<NN>/.

Variants (normalization statistics fitted on TRAIN recordings only):
  baseline_rerun : original global per-mel normalization (sanity check vs runs/baseline)
  V1_pos_norm    : per (position, view, mel) mean/std  -> removes position-wise common level
  V2_common_sub  : per (position, view, mel, frame) mean, per (position, view, mel) std
                   -> subtracts the common angle-dependent pattern (frame = angle inside ROI)
  V3_relative    : V2 + subtract each recording's own mean spectrum over the 3 positions
                   -> keeps only what is louder/softer in one direction relative to the others
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG))
from ev3dar.models import SoundTypeCNN
from ev3dar.metrics import classification_metrics
from ev3dar.engine import seed_all
from ev3dar.common import load_manifest, records_by_ids, read_json
from ev3dar.features import cache_path, feature_key

BASE = PKG / 'runs/baseline'; OUT = PKG / 'runs/exp_fullrot_20261116'
CFG = read_json(BASE / 'config.resolved.json')


def load_views(records):
    V, M, D, Y = [], [], [], []
    for r in records:
        with np.load(cache_path(BASE, r['uid']), allow_pickle=False) as z:
            assert str(z['feature_key'].item()) == feature_key(r, CFG), 'stale cache'
            V.append(z['id_views'].astype(np.float32)); M.append(z['id_masks'].astype(np.float32))
            D.append(z['view_durations'].astype(np.float32))
        Y.append([CFG['classes'].index(c) for c in r['classes']])
    return np.stack(V), np.stack(M), np.stack(D), np.array(Y, np.int64)


def fit_norm(V, M, variant):
    m = M[:, :, :, None, :]                                   # (n,3,2,1,T)
    if variant == 'baseline_rerun':
        s = (V * m).sum((0, 1, 2, 4)); q = (V * V * m).sum((0, 1, 2, 4)); n = m.sum((0, 1, 2, 4))
        mean = s / n; std = np.sqrt(np.maximum(q / n - mean ** 2, 1.0))
        return mean[None, None, :, None], std[None, None, :, None]
    s = (V * m).sum((0, 4)); q = (V * V * m).sum((0, 4)); n = np.maximum(m.sum((0, 4)), 1)
    mean = s / n; std = np.sqrt(np.maximum(q / n - mean ** 2, 1.0))          # (3,2,64)
    if variant == 'V1_pos_norm':
        return mean[..., None], std[..., None]
    if variant in ('V2_common_sub', 'V3_relative'):
        sf = (V * m).sum(0); nf = m.sum(0)                                     # (3,2,64,T)
        meanf = np.where(nf > 0, sf / np.maximum(nf, 1), mean[..., None])
        return meanf, std[..., None]
    raise ValueError(variant)


class DS(Dataset):
    def __init__(s, V, M, D, Y, mean, std, relative=False):
        m = M[:, :, :, None, :]
        Z = (V - mean) / std
        if relative:  # V3: subtract this recording's own mean spectrum over all 3 positions (valid frames)
            rec_mean = (Z * m).sum((1, 2, 4)) / np.maximum(m.sum((1, 2, 4)), 1)      # (n,64)
            Z = Z - rec_mean[:, None, None, :, None]
        s.V = Z * m; s.M, s.D, s.Y = M, D, Y
    def __len__(s): return len(s.Y)
    def __getitem__(s, i):
        return dict(views=torch.from_numpy(s.V[i].astype(np.float32)), masks=torch.from_numpy(s.M[i]),
                    durations=torch.from_numpy(s.D[i]), rpm=torch.zeros(3, 2), labels=torch.from_numpy(s.Y[i]))


def predict(model, loader, dev):
    model.eval(); P = []; Y = []
    with torch.no_grad():
        for b in loader:
            lo, _ = model(*(b[k].to(dev) for k in ['views', 'masks', 'durations', 'rpm']))
            P.append(lo.softmax(-1).cpu().numpy()); Y.append(b['labels'].numpy())
    return np.concatenate(P), np.concatenate(Y)


def run(variant, seed, epochs=None, init_offset=0, tag=''):
    dev = torch.device('cuda'); seed_all(seed + init_offset)  # split stays seed; init/shuffle seed may be offset
    recs = load_manifest(BASE); sp = read_json(BASE / f'splits/seed{seed}.json')
    data = {k: load_views(records_by_ids(recs, sp[k])) for k in ('train', 'validation', 'test')}
    mean, std = fit_norm(data['train'][0], data['train'][1], variant)
    ds = {k: DS(*v, mean, std, relative=(variant == 'V3_relative')) for k, v in data.items()}
    g = torch.Generator().manual_seed(seed + init_offset)
    tr = DataLoader(ds['train'], batch_size=CFG['batch_size'], shuffle=True, generator=g, num_workers=0)
    ld = {k: DataLoader(ds[k], batch_size=CFG['batch_size']) for k in ds}
    model = SoundTypeCNN(len(CFG['classes']), False).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'])
    best = -1e9; stale = 0; hist = []; best_state = None
    for ep in range(1, (epochs or CFG['epochs']) + 1):
        model.train(); tot = 0; n = 0; t0 = time.monotonic()
        for b in tr:
            opt.zero_grad(set_to_none=True)
            lo, _ = model(*(b[k].to(dev) for k in ['views', 'masks', 'durations', 'rpm']))
            loss = F.cross_entropy(lo.reshape(-1, 6), b['labels'].to(dev).reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss) * len(b['labels']); n += len(b['labels'])
        P, Y = predict(model, ld['validation'], dev)
        vm = classification_metrics(Y, P, CFG['classes'], CFG['fault_threshold'])
        hist.append(dict(epoch=ep, train_loss=tot / n, val_macro_f1=vm['type_macro_f1'],
                         val_fault_auc=vm['fault_roc_auc'], seconds=time.monotonic() - t0))
        if vm['type_macro_f1'] > best:
            best = vm['type_macro_f1']; stale = 0; best_ep = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= CFG['patience']: break
    model.load_state_dict(best_state)
    od = OUT / (variant + tag) / f'seed{seed}'; od.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=best_state, variant=variant, seed=seed, epoch=best_ep), od / 'id_best.pt')
    json.dump(hist, open(od / 'id_history.json', 'w'), indent=1)
    rows = []
    for part in ('train', 'validation', 'test'):
        P, Y = predict(model, ld[part], dev)
        ids = sp[part]
        for i, u in enumerate(ids):
            for k, pos in enumerate(CFG['positions']):
                rows.append(dict(split=part, uid=u, position=pos, gt=CFG['classes'][Y[i, k]],
                                 **{'p_' + c: float(P[i, k, j]) for j, c in enumerate(CFG['classes'])}))
    json.dump(rows, open(od / 'predictions.json', 'w'))
    print(variant + tag, seed, 'epochs', len(hist), 'best', best_ep, 'val macroF1', round(best, 3), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--variants', nargs='+')
    ap.add_argument('--epochs', type=int); ap.add_argument('--init_offset', type=int, default=0); ap.add_argument('--tag', default='')
    a = ap.parse_args()
    for v in a.variants:
        for s in (42, 43, 44): run(v, s, a.epochs, a.init_offset, a.tag)
