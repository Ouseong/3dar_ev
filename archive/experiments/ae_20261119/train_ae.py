"""Normal-only convolutional autoencoder for per-ROI fault detection (user request, 2026-11-19).

Unit: one angular ROI of one recording (FRONT 135-225, RIGHT 225-315, REAR 315-360 + 0-45).
REAR is encoded as two separate views (no spliced audio across 360/0), per CLAUDE.md item 5.
Training uses ONLY Healthy ROIs from the train recordings of the existing recording-level
8:1:1 split (seed42/43/44). Early stopping uses validation Healthy reconstruction loss only.
The fault threshold is chosen on validation ROIs (Healthy + fault), then applied once to test.
Anomaly score = duration-weighted masked MSE of the ROI's views.
This replaces CLAUDE.md item 3 ("no normal-only AE") by explicit user instruction.

Usage: python train_ae.py --seeds 42 43 44 [--epochs 100] [--out runs/ae_normal_20261119]
"""
from pathlib import Path
import argparse, json, sys, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_config, load_manifest, records_by_ids, read_json, write_json, write_csv
from ev3dar.features import cache_path, feature_key
from ev3dar.engine import seed_all, GpuThrottle

BASE = PROJECT / 'runs/baseline'          # prepared manifest + splits (read-only use)
CACHE = BASE                              # feature cache root; replaced by --cache for narrower ROIs
PAD_T = 104                               # 100 frames padded to a multiple of 8 for the conv stack
CENTERS = {'FRONT': 180.0, 'RIGHT': 270.0, 'REAR': 360.0}   # REAR centre 0 deg written as 360


def roi_config(cfg, width):
    """Same centres as the baseline ROIs, narrower width. REAR stays split at 360/0."""
    cfg = dict(cfg)
    cfg['rois'] = {p: [c - width / 2, c + width / 2] for p, c in CENTERS.items()}
    return cfg


def load_rois(records, cfg):
    """Return per-ROI arrays: views (N,2,64,100), masks (N,2,100), dur (N,2), meta list."""
    V, M, D, meta = [], [], [], []
    for r in records:
        with np.load(cache_path(CACHE, r['uid']), allow_pickle=False) as z:
            if str(z['feature_key'].item()) != feature_key(r, cfg):
                raise ValueError('Stale feature cache for ' + r['uid'])
            views, masks, durs = z['id_views'], z['id_masks'], z['view_durations']
        for pi, pos in enumerate(cfg['positions']):
            V.append(views[pi]); M.append(masks[pi]); D.append(durs[pi])
            meta.append(dict(uid=r['uid'], position=pos, sound_class=r['classes'][pi],
                             fault=int(r['classes'][pi] != 'Healthy'), band=r['band'],
                             fault_count=r['fault_count']))
    return np.stack(V).astype(np.float32), np.stack(M).astype(np.float32), np.stack(D).astype(np.float32), meta


def fit_stats(V, M):
    """Per-mel mean/std over valid frames of TRAIN HEALTHY views only."""
    m = M[:, :, None, :]
    n = m.sum(axis=(0, 1, 3))
    mean = (V * m).sum(axis=(0, 1, 3)) / n
    var = ((V - mean[None, None, :, None]) ** 2 * m).sum(axis=(0, 1, 3)) / n
    return mean.astype(np.float32), np.sqrt(np.maximum(var, 1e-6)).astype(np.float32)


def to_views(V, M, D, mean, std):
    """Flatten ROI views to independent view samples (only views that exist)."""
    x = (V - mean[None, None, :, None]) / std[None, None, :, None] * M[:, :, None, :]
    x = np.pad(x, ((0, 0), (0, 0), (0, 0), (0, PAD_T - x.shape[-1])))
    m = np.pad(M, ((0, 0), (0, 0), (0, PAD_T - M.shape[-1])))
    roi_idx, view_x, view_m, view_w = [], [], [], []
    for i in range(len(V)):
        for v in range(V.shape[1]):
            if M[i, v].sum() > 0:
                roi_idx.append(i); view_x.append(x[i, v]); view_m.append(m[i, v]); view_w.append(D[i, v])
    return (torch.tensor(np.stack(view_x))[:, None], torch.tensor(np.stack(view_m))[:, None, None, :],
            np.array(view_w, np.float32), np.array(roi_idx))


class ConvAE(nn.Module):
    """Encoder: 1x64x104 -> 128x8x13 -> latent 64. Decoder mirrors it back to 1x64x104."""
    def __init__(self, latent=64):
        super().__init__()
        def down(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        def up(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.enc = nn.Sequential(down(1, 32), down(32, 64), down(64, 128))
        self.to_z = nn.Linear(128 * 8 * 13, latent)
        self.from_z = nn.Linear(latent, 128 * 8 * 13)
        self.dec = nn.Sequential(up(128, 64), up(64, 32), nn.ConvTranspose2d(32, 1, 4, 2, 1))

    def forward(self, x):
        z = self.to_z(self.enc(x).flatten(1))
        return self.dec(self.from_z(z).view(-1, 128, 8, 13)), z


def view_errors(model, X, Mk, device, bs=256):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            x = X[i:i+bs].to(device); m = Mk[i:i+bs].to(device)
            rec, _ = model(x)
            err = ((rec - x) ** 2 * m).sum((1, 2, 3)) / (m.sum((1, 2, 3)) * x.shape[2])
            out.append(err.cpu().numpy())
    return np.concatenate(out)


def roi_scores(err, w, roi_idx, n):
    s = np.zeros(n); t = np.zeros(n)
    np.add.at(s, roi_idx, err * w); np.add.at(t, roi_idx, w)
    return s / t


def best_threshold(scores, y):
    """Maximize balanced accuracy on validation; ties -> lower threshold."""
    cands = np.unique(scores); best = (-1, None)
    for c in cands:
        p = scores >= c
        tpr = (p & (y == 1)).sum() / max((y == 1).sum(), 1); tnr = (~p & (y == 0)).sum() / max((y == 0).sum(), 1)
        ba = (tpr + tnr) / 2
        if ba > best[0]: best = (ba, float(c))
    return best[1]


def summarize(scores, y, thr, meta, cfg):
    pred = (scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    res = dict(n=len(y), n_healthy=int((y == 0).sum()), n_fault=int((y == 1).sum()), threshold=thr,
               roc_auc=float(roc_auc_score(y, scores)), average_precision=float(average_precision_score(y, scores)),
               accuracy=float((pred == y).mean()), balanced_accuracy=float((tp/(tp+fn) + tn/(tn+fp)) / 2),
               precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)),
               specificity=float(tn / max(tn + fp, 1)), f1=float(f1_score(y, pred, zero_division=0)),
               confusion_matrix_tn_fp_fn_tp=[int(tn), int(fp), int(fn), int(tp)])
    res['per_position'] = {}
    for pos in cfg['positions']:
        k = np.array([m['position'] == pos for m in meta])
        res['per_position'][pos] = dict(n=int(k.sum()), roc_auc=float(roc_auc_score(y[k], scores[k])),
                                        accuracy=float((pred[k] == y[k]).mean()))
    res['detection_rate_per_class'] = {}
    for c in cfg['classes']:
        k = np.array([m['sound_class'] == c for m in meta])
        res['detection_rate_per_class'][c] = dict(n=int(k.sum()), flagged_as_fault=float(pred[k].mean()),
                                                  mean_score=float(scores[k].mean()))
    return res


def run_seed(seed, cfg, records, args, device, out):
    seed_all(seed); t0 = time.time()
    split = read_json(BASE / f'splits/seed{seed}.json')
    sets = {k: records_by_ids(records, split[k]) for k in ['train', 'validation', 'test']}
    data = {k: load_rois(v, cfg) for k, v in sets.items()}
    # Train on Healthy ROIs only.
    Vtr, Mtr, Dtr, meta_tr = data['train']
    h = np.array([m['fault'] == 0 for m in meta_tr])
    mean, std = fit_stats(Vtr[h], Mtr[h])
    Xtr, Mktr, _, _ = to_views(Vtr[h], Mtr[h], Dtr[h], mean, std)
    Vva, Mva, Dva, meta_va = data['validation']
    yva = np.array([m['fault'] for m in meta_va])
    Xva, Mkva, Wva, Iva = to_views(Vva, Mva, Dva, mean, std)
    hv = yva[Iva] == 0                           # healthy validation views for early stopping

    model = ConvAE(args.latent).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
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
        val_healthy = float(view_errors(model, Xva[hv], Mkva[hv], device).mean())
        history.append(dict(epoch=ep, train_loss=tot / len(Xtr), val_healthy_recon=val_healthy))
        print(json.dumps(dict(seed=seed, **history[-1])), flush=True)
        if val_healthy < best:
            best, stale = val_healthy, 0
            torch.save(dict(state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                            epoch=ep, mean=mean, std=std, seed=seed, args=vars(args)), sd / 'ae_best.pt')
        else:
            stale += 1
            if stale >= args.patience: break
    write_json(sd / 'history.json', history)

    ck = torch.load(sd / 'ae_best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(ck['state_dict'])
    s_va = roi_scores(view_errors(model, Xva, Mkva, device), Wva, Iva, len(meta_va))
    thr = best_threshold(s_va, yva)
    Vte, Mte, Dte, meta_te = data['test']
    yte = np.array([m['fault'] for m in meta_te])
    Xte, Mkte, Wte, Ite = to_views(Vte, Mte, Dte, mean, std)
    s_te = roi_scores(view_errors(model, Xte, Mkte, device), Wte, Ite, len(meta_te))
    result = dict(seed=seed, best_epoch=ck['epoch'], epochs_run=len(history),
                  n_train_healthy_rois=int(h.sum()), validation=summarize(s_va, yva, thr, meta_va, cfg),
                  test=summarize(s_te, yte, thr, meta_te, cfg), seconds=time.time() - t0)
    write_json(sd / 'metrics.json', result)
    write_csv(sd / 'test_predictions.csv', [dict(**m, score=float(s), predicted_fault=int(s >= thr))
                                            for m, s in zip(meta_te, s_te)])
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=15)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--latent', type=int, default=64)
    ap.add_argument('--out', default=str(PROJECT / 'runs/ae_normal_20261119'))
    ap.add_argument('--roi-width', type=float, default=90.0)
    ap.add_argument('--cache', default=None, help='feature cache root (contains cache/); default runs/baseline')
    args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback.')
    global CACHE
    cfg = load_config(BASE / 'config.resolved.json'); records = load_manifest(BASE)
    if args.roi_width != 90.0:
        if not args.cache: raise ValueError('Narrower ROIs need their own --cache (see build_cache_roi.py).')
        cfg = roi_config(cfg, args.roi_width)
    if args.cache: CACHE = Path(args.cache)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = [run_seed(s, cfg, records, args, torch.device('cuda'), out) for s in args.seeds]
    keys = ['roc_auc', 'average_precision', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'specificity', 'f1']
    summary = {k: dict(mean=float(np.mean([r['test'][k] for r in results])),
                       std=float(np.std([r['test'][k] for r in results]))) for k in keys}
    write_json(out / 'summary.json', dict(seeds=args.seeds, test=summary, args=vars(args)))
    print(json.dumps(dict(summary=summary), indent=1))


if __name__ == '__main__':
    main()
