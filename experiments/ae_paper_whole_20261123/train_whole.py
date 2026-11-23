"""Paper reproduction on whole recordings (Kunte et al., 'Domain Knowledge Informed Approach for Anomaly
Detection of EV Interior Sounds'). User request 2026-11-23.

Unit = one whole recording (~4 s, 3 speakers), input 1 x 1536 freq x 384 time/RPM (build_cache_whole.py).
Label = recording-level: Healthy if fault_count == 0 (300 recordings), Fault otherwise (900). No ROI / position.
Split = existing recording-level runs/baseline/splits/seed{42,43,44}.json (Healthy train 240 / val 30 / test 30).

Paper procedure reproduced:
  - AE (Fig. 10): n encoder blocks Conv3x3(f)-BN-ReLU-MaxPool2, n-1 decoder blocks ConvT-Conv-BN-ReLU, ConvT-Conv-Sigmoid.
  - Table 5: Adam 1e-4, batch 8, <= 100 epochs, MSE. Train on Healthy only; early stop on Healthy-val reconstruction.
  - Input scaling: clip dB to a 120 dB range, /120 (dataset-wide). Ours: clip(-120..0 dBFS)/120 + 1.
  - Proxy anomalies (Sec. 4.2, Table 4) on Healthy validation, split into 3 equal subsets, one perturbation each:
      add_rpm  : +U[0.1,0.25] on 4 adjacent RPM(time) columns
      add_freq : +U[0.1,0.25] on 4 adjacent frequency rows
      add_order: +U[0.1,0.25] on the line f = order * RPM / 60, width 15 Hz, order from {2, 2.5, ..., 40}
    Values clipped to [0, 1] after adding (not stated in the paper). Perturbations are fixed per seed
    (same proxies for every architecture). RPM per column = estimated interval, linear in time (not measured).
  - Model selection scores stored per run: val_RE (Healthy-val MSE), proxy AUROC (Healthy val vs proxies),
    and, because we DO have real fault labels, real-val AUROC (Healthy val vs Fault val) for comparison.
  - Test AUROC (Healthy test vs Fault test) is reported for every run; selection never looks at it.
Anomaly score = mean squared reconstruction error over the whole spectrogram.

Usage: python train_whole.py --blocks 6 --filters 8 --seed 42
       python train_whole.py --sweep --seed 42          (n = 1..7 x f = 4,8,16,32; skips finished runs)
"""
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(HERE))
from ev3dar.common import load_manifest, read_json, records_by_ids, write_json, write_csv
from ev3dar.features import cache_path
from ev3dar.engine import seed_all, GpuThrottle
from build_cache_whole import OUT as CACHE_ROOT, whole_key, SPEC

BASE = PROJECT / 'runs/baseline'
RUNS = PROJECT / 'runs/ae_paper_whole_20261123/runs'
PROXY = dict(a=0.1, b=0.25, n_rpm=4, n_freq=4, t_order_hz=15.0, c_order=2.0, d_order=40.0)
BIN_HZ = 48000 / SPEC['n_fft']


class PaperAE(nn.Module):
    def __init__(self, n, f):
        super().__init__()
        enc, c = [], 1
        for _ in range(n):
            enc += [nn.Conv2d(c, f, 3, 1, 1), nn.BatchNorm2d(f), nn.ReLU(), nn.MaxPool2d(2, 2)]; c = f
        dec = []
        for _ in range(n - 1):
            dec += [nn.ConvTranspose2d(f, f, 2, 2), nn.Conv2d(f, f, 3, 1, 1), nn.BatchNorm2d(f), nn.ReLU()]
        dec += [nn.ConvTranspose2d(f, f, 2, 2), nn.Conv2d(f, 1, 3, 1, 1), nn.Sigmoid()]
        self.enc, self.dec = nn.Sequential(*enc), nn.Sequential(*dec)

    def forward(self, x):
        return self.dec(self.enc(x))


def load(records):
    X, rpm = [], []
    for r in records:
        with np.load(cache_path(CACHE_ROOT, r['uid']), allow_pickle=False) as z:
            if str(z['feature_key'].item()) != whole_key(r): raise ValueError('Stale whole cache ' + r['uid'])
            X.append(z['spec']); rpm.append((float(z['rpm0']), float(z['rpm1'])))
    X = (np.clip(np.stack(X).astype(np.float32), -120.0, 0.0) / 120.0 + 1.0)[:, None]
    return torch.from_numpy(X), np.array(rpm)


def make_proxies(Xh, rpm, seed):
    """Paper Sec. 4.2: divide Healthy validation into 3 subsets, perturb each with one type."""
    rng = np.random.default_rng(10_000 + seed)
    idx = rng.permutation(len(Xh)); parts = np.array_split(idx, 3)
    out, kinds = [], []
    F_, T_ = Xh.shape[2], Xh.shape[3]
    for kind, part in zip(['add_rpm', 'add_freq', 'add_order'], parts):
        for i in part:
            x = Xh[i, 0].numpy().copy(); d = rng.uniform(PROXY['a'], PROXY['b'])
            if kind == 'add_rpm':
                r = rng.integers(0, T_ - PROXY['n_rpm'] + 1); x[:, r:r + PROXY['n_rpm']] += d
            elif kind == 'add_freq':
                f = rng.integers(0, F_ - PROXY['n_freq'] + 1); x[f:f + PROXY['n_freq'], :] += d
            else:
                orders = np.arange(PROXY['c_order'], PROXY['d_order'] + 1e-9, 0.5)
                o = rng.choice(orders)
                rpm_col = np.linspace(rpm[i, 0], rpm[i, 1], T_)
                fc = o * rpm_col / 60.0                                   # Hz per column
                freqs = np.arange(F_) * BIN_HZ
                band = np.abs(freqs[:, None] - fc[None, :]) <= PROXY['t_order_hz'] / 2
                band |= (np.abs(freqs[:, None] - fc[None, :]) <= BIN_HZ / 2)   # >= 1 bin per column
                x[band] += d
            out.append(np.clip(x, 0.0, 1.0)); kinds.append(kind)
    return torch.from_numpy(np.stack(out))[:, None], kinds


def recon_errors(model, X, device, bs=8):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = X[i:i + bs].to(device); out.append(((model(x) - x) ** 2).mean((1, 2, 3)).cpu().numpy())
    return np.concatenate(out)


def auc(neg, pos):
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def get_data(seed, records):
    split = read_json(BASE / f'splits/seed{seed}.json'); D = {}
    for k in ['train', 'validation', 'test']:
        recs = records_by_ids(records, split[k]); X, rpm = load(recs)
        y = np.array([int(r['fault_count'] > 0) for r in recs])
        D[k] = dict(recs=recs, X=X, rpm=rpm, y=y)
    vh = D['validation']['y'] == 0
    D['proxy_X'], D['proxy_kind'] = make_proxies(D['validation']['X'][vh], D['validation']['rpm'][vh], seed)
    return D


def run(n, f, seed, D, args, device):
    sd = RUNS / f'seed{seed}' / f'n{n}_f{f}'
    if (sd / 'metrics.json').exists() and not args.force: return read_json(sd / 'metrics.json')
    sd.mkdir(parents=True, exist_ok=True)
    seed_all(seed); t0 = time.time()
    tr = D['train']; Xtr = tr['X'][tr['y'] == 0]
    va = D['validation']; Xvh = va['X'][va['y'] == 0]
    model = PaperAE(n, f).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(seed); throttle = GpuThrottle()
    best, stale, hist = float('inf'), 0, []
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(Xtr), generator=g); tot = 0.0; throttle.reset()
        for i in range(0, len(perm), args.batch):
            x = Xtr[perm[i:i + args.batch]].to(device)
            loss = ((model(x) - x) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss) * len(x); throttle.step()
        vre = float(recon_errors(model, Xvh, device).mean())
        hist.append(dict(epoch=ep, train_loss=tot / len(Xtr), val_healthy_recon=vre))
        print(json.dumps(dict(n=n, f=f, seed=seed, **hist[-1])), flush=True)
        if vre < best:
            best, stale = vre, 0
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, sd / 'ae_best.pt')
        else:
            stale += 1
            if stale >= args.patience: break
    write_json(sd / 'history.json', hist)
    model.load_state_dict(torch.load(sd / 'ae_best.pt', map_location='cpu'))
    e_vh = recon_errors(model, Xvh, device); e_vf = recon_errors(model, va['X'][va['y'] == 1], device)
    e_px = recon_errors(model, D['proxy_X'], device)
    te = D['test']; e_te = recon_errors(model, te['X'], device)
    e_th, e_tf = e_te[te['y'] == 0], e_te[te['y'] == 1]
    kinds = np.array(D['proxy_kind'])
    res = dict(n=n, f=f, seed=seed, n_params=sum(p.numel() for p in model.parameters()),
               best_epoch=int(np.argmin([h['val_healthy_recon'] for h in hist]) + 1), epochs_run=len(hist),
               val_RE=float(e_vh.mean()), proxy_auroc=auc(e_vh, e_px),
               proxy_auroc_per_type={k: auc(e_vh, e_px[kinds == k]) for k in ['add_rpm', 'add_freq', 'add_order']},
               real_val_auroc=auc(e_vh, e_vf), test_auroc=auc(e_th, e_tf),
               test_auroc_by_fault_count={str(c): auc(e_th, e_te[np.array([r['fault_count'] == c for r in te['recs']])])
                                          for c in [1, 2, 3]},
               n_train_healthy=int(len(Xtr)), n_val_healthy=int(len(Xvh)), n_val_fault=int((va['y'] == 1).sum()),
               n_proxy=int(len(e_px)), n_test_healthy=int(len(e_th)), n_test_fault=int(len(e_tf)),
               seconds=time.time() - t0)
    write_json(sd / 'metrics.json', res)
    write_csv(sd / 'test_scores.csv', [dict(uid=r['uid'], fault_count=r['fault_count'], classes='|'.join(r['classes']),
                                            band=r['band'], score=float(s)) for r, s in zip(te['recs'], e_te)])
    (sd / 'ae_best.pt').unlink() if args.drop_ckpt else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', type=int, nargs='+', default=[6]); ap.add_argument('--filters', type=int, nargs='+', default=[8])
    ap.add_argument('--sweep', action='store_true', help='n = 1..7 x f = 4, 8, 16, 32')
    ap.add_argument('--seed', type=int, nargs='+', default=[42])
    ap.add_argument('--epochs', type=int, default=100); ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--batch', type=int, default=8); ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--force', action='store_true'); ap.add_argument('--drop-ckpt', action='store_true')
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback.')
    configs = [(n, f) for n in range(1, 8) for f in (4, 8, 16, 32)] if args.sweep else \
              [(n, f) for n in args.blocks for f in args.filters]
    records = load_manifest(BASE); device = torch.device('cuda')
    for seed in args.seed:
        D = get_data(seed, records)
        for n, f in configs:
            r = run(n, f, seed, D, args, device)
            print('RESULT', json.dumps({k: r[k] for k in ['n', 'f', 'seed', 'val_RE', 'proxy_auroc', 'real_val_auroc', 'test_auroc']}), flush=True)


if __name__ == '__main__':
    main()
