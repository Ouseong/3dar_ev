"""Normal-only AE with the architecture of Kunte et al. (KU Leuven, 'A Domain Knowledge Informed
Approach for Anomaly Detection of EV Interior Sounds', Fig. 10 / Table 5). User request 2026-11-23.

Paper architecture (n blocks, f filters):
  Encoder x n      : Conv2d(f,k3,s1,p1) -> BatchNorm2d -> ReLU -> MaxPool2d(2,2)
  Decoder x (n-1)  : ConvTranspose2d(f) -> Conv2d(f,k3,s1,p1) -> BatchNorm2d -> ReLU
  Output           : ConvTranspose2d(f) -> Conv2d(->1 channel) -> Sigmoid
  No dense bottleneck; latent = f x H/2^n x W/2^n. Paper's selected model: n=6, f=8.
  Not stated in the paper (our choices): ConvTranspose2d kernel 2 stride 2 (to undo MaxPool),
  last Conv2d outputs 1 channel (the figure says 'f filters', but the output is one spectrogram).
Paper hyper-parameters: Adam lr 1e-4, batch 8, max 100 epochs, MSE.
Paper input scaling: clip 0..120 dB then /120 (dataset-wide, not per sample). Our features are dBFS
  in [-120, ~-19], so we use clip(x, -120, 0)/120 + 1.
Paper model selection uses synthetic proxy anomalies; we do NOT have those, so this script only
  trains given (n, f). Early stopping on validation Healthy reconstruction error (patience).

Everything else (ROIs, splits, Healthy-only training, masked MSE, val threshold, metrics, outputs)
is reused unchanged from experiments/ae_20261119/train_ae.py.

Usage: python train_ae_paper.py --blocks 6 --filters 8 --seeds 42 43 44 [--feature mel|stft] [--roi-width 60]
"""
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(PROJECT / 'experiments/ae_20261119')); sys.path.insert(0, str(HERE))
import train_ae as base                                   # noqa: E402  (reuse data/eval code)
from ev3dar.common import load_config, load_manifest, read_json, records_by_ids, write_json, write_csv
from ev3dar.engine import seed_all, GpuThrottle

PAD_T = 128      # 100 frames -> 128 so that up to 2^7 pooling divides it (masked out of the loss)
STFT_CACHE = PROJECT / 'runs/ae_paper_20261123/stft_cache'
STFT_CACHE_DIR = STFT_CACHE   # set in main() for narrower ROIs


def load_rois_stft(records, cfg):
    """Like base.load_rois, but the linear-STFT cache from build_cache_stft.py (1536 freq rows)."""
    from ev3dar.features import cache_path
    from build_cache_stft import stft_key
    V, M, D, meta = [], [], [], []
    for r in records:
        with np.load(cache_path(STFT_CACHE_DIR, r['uid']), allow_pickle=False) as z:
            if str(z['feature_key'].item()) != stft_key(r, cfg): raise ValueError('Stale STFT cache for ' + r['uid'])
            views, masks, durs = z['stft_views'], z['id_masks'], z['view_durations']
        for pi, pos in enumerate(cfg['positions']):
            V.append(views[pi]); M.append(masks[pi]); D.append(durs[pi])
            meta.append(dict(uid=r['uid'], position=pos, sound_class=r['classes'][pi],
                             fault=int(r['classes'][pi] != 'Healthy'), band=r['band'], fault_count=r['fault_count']))
    return np.stack(V), np.stack(M).astype(np.float32), np.stack(D).astype(np.float32), meta


def scale(V):
    return (np.clip(V, -120.0, 0.0) / 120.0 + 1.0).astype(np.float32)


def to_views(V, M, D):
    """Like base.to_views, but paper scaling instead of z-score; padded frames = 0, masked."""
    x = scale(V.astype(np.float32)) * M[:, :, None, :]
    x = np.pad(x, ((0, 0), (0, 0), (0, 0), (0, PAD_T - x.shape[-1])))
    m = np.pad(M, ((0, 0), (0, 0), (0, PAD_T - M.shape[-1])))
    roi_idx, vx, vm, vw = [], [], [], []
    for i in range(len(V)):
        for v in range(V.shape[1]):
            if M[i, v].sum() > 0:
                roi_idx.append(i); vx.append(x[i, v]); vm.append(m[i, v]); vw.append(D[i, v])
    return (torch.tensor(np.stack(vx))[:, None], torch.tensor(np.stack(vm))[:, None, None, :],
            np.array(vw, np.float32), np.array(roi_idx))


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
        z = self.enc(x)
        return self.dec(z), z


def run_seed(seed, cfg, records, args, device, out):
    seed_all(seed); t0 = time.time()
    split = read_json(base.BASE / f'splits/seed{seed}.json')
    loader = load_rois_stft if args.feature == 'stft' else base.load_rois
    data = {k: loader(records_by_ids(records, split[k]), cfg) for k in ['train', 'validation', 'test']}
    Vtr, Mtr, Dtr, meta_tr = data['train']
    h = np.array([m['fault'] == 0 for m in meta_tr])
    Xtr, Mktr, _, _ = to_views(Vtr[h], Mtr[h], Dtr[h])
    Vva, Mva, Dva, meta_va = data['validation']
    yva = np.array([m['fault'] for m in meta_va])
    Xva, Mkva, Wva, Iva = to_views(Vva, Mva, Dva)
    hv = yva[Iva] == 0

    model = PaperAE(args.blocks, args.filters).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(seed); throttle = GpuThrottle()
    best, stale, history = float('inf'), 0, []
    sd = out / f'seed{seed}'; sd.mkdir(parents=True, exist_ok=True)
    for ep in range(1, args.epochs + 1):
        model.train(); perm = torch.randperm(len(Xtr), generator=g); tot = 0.0; throttle.reset()
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i+args.batch]; x = Xtr[idx].to(device); m = Mktr[idx].to(device)
            rec, _ = model(x)
            loss = ((rec - x) ** 2 * m).sum() / (m.sum() * x.shape[2])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += float(loss) * len(idx); throttle.step()
        val_h = float(base.view_errors(model, Xva[hv], Mkva[hv], device, bs=args.eval_bs).mean())
        history.append(dict(epoch=ep, train_loss=tot / len(Xtr), val_healthy_recon=val_h))
        print(json.dumps(dict(seed=seed, **history[-1])), flush=True)
        if val_h < best:
            best, stale = val_h, 0
            torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            epoch=ep, seed=seed, args=vars(args)), sd / 'ae_best.pt')
        else:
            stale += 1
            if stale >= args.patience: break
    write_json(sd / 'history.json', history)

    ck = torch.load(sd / 'ae_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'])
    # Reconstruction check: AE vs. a no-learning baseline that copies each view's own time-averaged spectrum.
    def own_avg_err(X, Mk):
        avg = (X * Mk).sum(3, keepdim=True) / Mk.sum(3, keepdim=True).clamp_min(1)
        return (((avg - X) ** 2 * Mk).sum((1, 2, 3)) / (Mk.sum((1, 2, 3)) * X.shape[2])).numpy()
    recon_check = dict(val_healthy_ae=float(base.view_errors(model, Xva[hv], Mkva[hv], device, bs=args.eval_bs).mean()),
                       val_healthy_own_avg_spectrum=float(own_avg_err(Xva[hv], Mkva[hv]).mean()))
    recon_check['ae_over_baseline'] = recon_check['val_healthy_ae'] / recon_check['val_healthy_own_avg_spectrum']
    s_va = base.roi_scores(base.view_errors(model, Xva, Mkva, device, bs=args.eval_bs), Wva, Iva, len(meta_va))
    thr = base.best_threshold(s_va, yva)
    Vte, Mte, Dte, meta_te = data['test']
    yte = np.array([m['fault'] for m in meta_te])
    Xte, Mkte, Wte, Ite = to_views(Vte, Mte, Dte)
    s_te = base.roi_scores(base.view_errors(model, Xte, Mkte, device, bs=args.eval_bs), Wte, Ite, len(meta_te))
    result = dict(seed=seed, feature=args.feature, blocks=args.blocks, filters=args.filters, best_epoch=ck['epoch'],
                  epochs_run=len(history), n_params=sum(p.numel() for p in model.parameters()),
                  n_train_healthy_rois=int(h.sum()), validation=base.summarize(s_va, yva, thr, meta_va, cfg),
                  test=base.summarize(s_te, yte, thr, meta_te, cfg), recon_check=recon_check, seconds=time.time() - t0)
    write_json(sd / 'metrics.json', result)
    write_csv(sd / 'test_predictions.csv', [dict(**m, score=float(s), predicted_fault=int(s >= thr))
                                            for m, s in zip(meta_te, s_te)])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', type=int, default=6)
    ap.add_argument('--filters', type=int, default=8)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--out', default=None)
    ap.add_argument('--feature', choices=['mel', 'stft'], default='mel', help='mel: 64-mel baseline cache; stft: 1536-bin linear STFT cache')
    ap.add_argument('--eval-bs', type=int, default=64)
    ap.add_argument('--roi-width', type=float, default=90.0, help='60: mel uses runs/ae_roi60_20261119, stft uses stft_cache_roi60')
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback.')
    nf = 1536 if args.feature == 'stft' else 64
    if nf % 2 ** args.blocks or PAD_T % 2 ** args.blocks: raise ValueError('freq rows and 128 frames must be divisible by 2^blocks.')
    cfg = load_config(base.BASE / 'config.resolved.json'); records = load_manifest(base.BASE)
    global STFT_CACHE_DIR
    if args.roi_width != 90:
        from build_cache_stft import out_dir
        cfg = base.roi_config(cfg, args.roi_width)
        base.CACHE = PROJECT / f'runs/ae_roi{int(args.roi_width)}_20261119'
        STFT_CACHE_DIR = out_dir(args.roi_width)
    out = Path(args.out or PROJECT / f'runs/ae_paper_20261123/{"stft_" if args.feature == "stft" else ""}{"" if args.roi_width == 90 else f"roi{int(args.roi_width)}_"}n{args.blocks}_f{args.filters}')
    out.mkdir(parents=True, exist_ok=True)
    results = [run_seed(s, cfg, records, args, torch.device('cuda'), out) for s in args.seeds]
    keys = ['roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'specificity', 'f1']
    summary = {k: dict(mean=float(np.mean([r['test'][k] for r in results])),
                       std=float(np.std([r['test'][k] for r in results]))) for k in keys}
    write_json(out / 'summary.json', dict(seeds=args.seeds, test=summary, args=vars(args)))
    print(json.dumps(dict(summary=summary), indent=1))


if __name__ == '__main__':
    main()
