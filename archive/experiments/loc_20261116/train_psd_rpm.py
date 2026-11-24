"""Experiment 3: spectrogram CNN vector + PSD vector + RPM -> FCN -> latent -> decoder.

Follows the agreed design: spectrogram -> CNN -> vector -> (concat extra info) -> FCN -> latent space -> decoder.
Only change to ev3dar.models.SoundTypeCNN: the FCN's first Linear accepts 128 + psd_dim + rpm_dim inputs.
Encoder, pooling, rear-segment merge, FCN widths (128 -> 64 latent), decoder, loss, optimizer, lr, batch,
epochs (50), patience, validation-macroF1 selection, splits: unchanged. ev3dar files are imported, not edited.

Variants: rpm (use_rpm=True, original mechanism), psd, psd_rpm.
RPM = ev3dar cache 'rpm' feature (estimated interval label, linear in time; NOT a measured RPM).
PSD normalized per band with TRAIN statistics only.
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np, torch
from torch import nn
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
PSD = np.load(OUT / 'psd_features.npz'); PSD_BY = {u: PSD['psd'][i] for i, u in enumerate(PSD['uids'])}


class SoundTypeCNNPlus(SoundTypeCNN):
    """SoundTypeCNN with extra vectors concatenated before the FCN (into the latent-space path)."""
    def __init__(self, classes=6, use_rpm=False, psd_dim=0):
        super().__init__(classes, use_rpm)
        self.psd_dim = psd_dim
        self.fcn[0] = nn.Linear(128 + (2 if use_rpm else 0) + psd_dim, 128)

    def forward(self, views, masks, durations, rpm, psd=None):
        b, p, v, m, t = views.shape                                   # identical to SoundTypeCNN.forward ...
        x = self.encoder(views.reshape(b * p * v, 1, m, t))
        valid = F.adaptive_avg_pool1d(masks.reshape(b * p * v, 1, t), x.shape[-1]).unsqueeze(2)
        z = (x * valid).sum((2, 3)) / (valid.sum((2, 3)) * x.shape[2]).clamp_min(1e-6)
        z = z.reshape(b, p, v, -1)
        weights = durations * (masks.sum(-1) > 0).to(durations.dtype)
        z = (z * weights.unsqueeze(-1)).sum(2) / weights.sum(2, keepdim=True).clamp_min(1e-6)
        if self.psd_dim: z = torch.cat([z, psd], dim=-1)              # ... plus PSD vector
        if self.use_rpm: z = torch.cat([z, rpm], dim=-1)              # ... and RPM (original mechanism)
        latent = self.fcn(z)
        return self.decoder(latent), latent


def load(records):
    V, M, D, R, Y, S = [], [], [], [], [], []
    for r in records:
        with np.load(cache_path(BASE, r['uid']), allow_pickle=False) as z:
            assert str(z['feature_key'].item()) == feature_key(r, CFG)
            V.append(z['id_views'].astype(np.float32)); M.append(z['id_masks'].astype(np.float32))
            D.append(z['view_durations'].astype(np.float32)); R.append(z['rpm'].astype(np.float32))
        Y.append([CFG['classes'].index(c) for c in r['classes']]); S.append(PSD_BY[r['uid']])
    return [np.stack(a) for a in (V, M, D, R)] + [np.array(Y, np.int64), np.stack(S).astype(np.float32)]


class DS(Dataset):
    def __init__(s, V, M, D, R, Y, S, norm):
        s.V = ((V - norm['im']) / norm['is']) * M[:, :, :, None, :]
        s.S = (S - norm['pm']) / norm['ps']; s.M, s.D, s.R, s.Y = M, D, R, Y
    def __len__(s): return len(s.Y)
    def __getitem__(s, i):
        return dict(views=torch.from_numpy(s.V[i]), masks=torch.from_numpy(s.M[i]), durations=torch.from_numpy(s.D[i]),
                    rpm=torch.from_numpy(s.R[i]), psd=torch.from_numpy(s.S[i]), labels=torch.from_numpy(s.Y[i]))


def fit_norm(V, M, S):
    m = M[:, :, :, None, :]
    s = (V * m).sum((0, 1, 2, 4)); q = (V * V * m).sum((0, 1, 2, 4)); n = m.sum((0, 1, 2, 4))
    im = s / n; isd = np.sqrt(np.maximum(q / n - im ** 2, 1.0))          # same as ev3dar global per-mel norm
    pm = S.mean((0, 1)); ps = np.maximum(S.std((0, 1)), 1e-3)            # per PSD band, train only
    return dict(im=im[None, None, :, None].astype(np.float32), **{'is': isd[None, None, :, None].astype(np.float32)},
                pm=pm.astype(np.float32), ps=ps.astype(np.float32))


def fwd(model, b, dev):
    return model(*(b[k].to(dev) for k in ['views', 'masks', 'durations', 'rpm', 'psd']))


def predict(model, loader, dev):
    model.eval(); P = []; Y = []
    with torch.no_grad():
        for b in loader:
            lo, _ = fwd(model, b, dev); P.append(lo.softmax(-1).cpu().numpy()); Y.append(b['labels'].numpy())
    return np.concatenate(P), np.concatenate(Y)


def run(variant, seed, init_offset=0, tag=''):
    use_rpm = 'rpm' in variant; psd_dim = 64 if 'psd' in variant else 0
    dev = torch.device('cuda'); seed_all(seed + init_offset)
    recs = load_manifest(BASE); sp = read_json(BASE / f'splits/seed{seed}.json')
    data = {k: load(records_by_ids(recs, sp[k])) for k in ('train', 'validation', 'test')}
    norm = fit_norm(data['train'][0], data['train'][1], data['train'][5])
    ds = {k: DS(*v, norm) for k, v in data.items()}
    g = torch.Generator().manual_seed(seed + init_offset)
    tr = DataLoader(ds['train'], batch_size=CFG['batch_size'], shuffle=True, generator=g)
    ld = {k: DataLoader(ds[k], batch_size=CFG['batch_size']) for k in ds}
    model = SoundTypeCNNPlus(6, use_rpm, psd_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['learning_rate'], weight_decay=CFG['weight_decay'])
    best = -1e9; stale = 0; hist = []
    for ep in range(1, CFG['epochs'] + 1):
        model.train(); tot = 0; n = 0; t0 = time.monotonic()
        for b in tr:
            opt.zero_grad(set_to_none=True); lo, _ = fwd(model, b, dev)
            loss = F.cross_entropy(lo.reshape(-1, 6), b['labels'].to(dev).reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.detach()) * len(b['labels']); n += len(b['labels'])
        P, Y = predict(model, ld['validation'], dev)
        vm = classification_metrics(Y, P, CFG['classes'], CFG['fault_threshold'])
        hist.append(dict(epoch=ep, train_loss=tot / n, val_macro_f1=vm['type_macro_f1'], val_fault_auc=vm['fault_roc_auc'],
                         seconds=time.monotonic() - t0))
        if vm['type_macro_f1'] > best:
            best = vm['type_macro_f1']; stale = 0; best_ep = ep
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if stale >= CFG['patience']: break
    model.load_state_dict(state)
    od = OUT / (variant + tag) / f'seed{seed}'; od.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=state, variant=variant, seed=seed, epoch=best_ep), od / 'id_best.pt')
    json.dump(hist, open(od / 'id_history.json', 'w'), indent=1)
    rows = []
    for part in ('train', 'validation', 'test'):
        P, Y = predict(model, ld[part], dev)
        for i, u in enumerate(sp[part]):
            for k, pos in enumerate(CFG['positions']):
                rows.append(dict(split=part, uid=u, position=pos, gt=CFG['classes'][Y[i, k]],
                                 **{'p_' + c: float(P[i, k, j]) for j, c in enumerate(CFG['classes'])}))
    json.dump(rows, open(od / 'predictions.json', 'w'))
    print(variant + tag, seed, 'epochs', len(hist), 'best', best_ep, 'val macroF1', round(best, 3), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--variants', nargs='+')
    ap.add_argument('--init_offset', type=int, default=0); ap.add_argument('--tag', default=''); a = ap.parse_args()
    for v in a.variants:
        for s in (42, 43, 44): run(v, s, a.init_offset, a.tag)
