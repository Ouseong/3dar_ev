"""FaultBandNet: a neural network on fault-band x mic-angle maps (inputs from build_inputs.py). 2026-11-24.
Input : wind 36x58, pwm 36x68, whine 36x51 (angle bins x frequency bins, level-free inside each recording).
Model : one 2-D CNN branch per map (3 conv layers, circular padding along the angle axis, pooling along frequency only)
        -> per-angle feature vectors (3 x 64 channels x 36 angles) -> two circular 1-D convolutions over angle
        -> fully connected -> 9 sigmoid outputs = 3 speaker positions (FRONT, REAR, RIGHT) x 3 fault types (Whine, PWM, Wind).
Labels: output (p, c) = 1 if the speaker at position p plays fault c. Imbalance / Modulation speakers are labelled 0
        (not detectable in the recordings, see experiments/ku_original_20261123).
Train : BCE with positive-class weights, AdamW 1e-3 (wd 1e-3), batch 32, <= 80 epochs, early stopping on validation AUC.
Protocols (all supervised, test recordings never used for training / model selection):
  cv5    : 5-fold grouped by the RIGHT speaker's source file (same protocol as the rule detector), 15 % of the training
           folds held out (grouped) as validation.
  strict : source-disjoint splits runs/strict_split_20261123/seed{42,43,44} (no source file of any speaker shared).
GPU duty ~50 %. Output: runs/fault_net_20261124/{cv5,strict}_probs.npy, summary.json
"""
from pathlib import Path
import sys, json, collections
import numpy as np, torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT))
from ev3dar.common import load_manifest, read_json, write_json
from ev3dar.engine import seed_all, GpuThrottle
OUT = PROJECT / 'runs/fault_net_20261124'; P3 = ['FRONT', 'REAR', 'RIGHT']; TY = ['Whine', 'PWM', 'Wind']
R = load_manifest(PROJECT / 'runs/baseline'); n = len(R); z = np.load(OUT / 'inputs.npz'); assert list(z['uid']) == [r['uid'] for r in R]
X = [torch.from_numpy(z[k].astype(np.float32))[:, None] for k in ('wind', 'pwm', 'whine')]
Y = np.array([[[float(r['classes'][p] == c) for c in TY] for p in range(3)] for r in R], np.float32)       # (n, pos, type)
fc = np.array([r['fault_count'] for r in R]); date = np.array([r['event_time'][5:10] for r in R]); grp = np.array([r['source_files'][2] for r in R])


class Branch(nn.Module):
    def __init__(s):
        super().__init__(); s.c = nn.ModuleList([nn.Conv2d(a, b, 3) for a, b in ((1, 16), (16, 32), (32, 32))]); s.n = nn.ModuleList([nn.GroupNorm(4, b) for b in (16, 32, 32)])
    def forward(s, x):
        for i, (c, g) in enumerate(zip(s.c, s.n)):
            x = F.gelu(g(c(F.pad(F.pad(x, (1, 1, 0, 0)), (0, 0, 1, 1), mode='circular'))))
            if i < 2: x = F.max_pool2d(x, (1, 2))
        return torch.cat([x.mean(3), x.amax(3)], 1)                       # (B, 64, 36 angles)


class FaultBandNet(nn.Module):
    def __init__(s):
        super().__init__(); s.b = nn.ModuleList([Branch() for _ in range(3)])
        s.a1 = nn.Conv1d(192, 64, 5, padding=2, padding_mode='circular'); s.a2 = nn.Conv1d(64, 64, 5, padding=2, padding_mode='circular')
        s.drop = nn.Dropout(.3); s.out = nn.Linear(64 * 36, 9)
    def forward(s, xs):
        h = torch.cat([b(x) for b, x in zip(s.b, xs)], 1); h = F.gelu(s.a2(s.drop(F.gelu(s.a1(h)))))
        return s.out(s.drop(h.flatten(1))).view(-1, 3, 3)


def rec_auc(P, idx):
    v = []
    for c in range(3):
        y = Y[idx][:, :, c].max(1)
        if 0 < y.mean() < 1: v.append(roc_auc_score(y, P[:, :, c].max(1)))
    return float(np.mean(v))


def fit_predict(tr, va, te, seed, dev):
    seed_all(seed); mu = [x[tr].mean() for x in X]; sd = [x[tr].std() for x in X]; Z = [(x - m) / s for x, m, s in zip(X, mu, sd)]
    Yt = torch.from_numpy(Y); pos = Y[tr].reshape(len(tr), -1).sum(0); pw = torch.tensor(np.clip((len(tr) - pos) / np.maximum(pos, 1), 1, 20), dtype=torch.float32, device=dev).view(1, 3, 3)
    model = FaultBandNet().to(dev); opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-3); g = torch.Generator().manual_seed(seed); thr = GpuThrottle()
    def predict(ix):
        model.eval(); o = []
        with torch.no_grad():
            for i in range(0, len(ix), 128): b = ix[i:i + 128]; o.append(torch.sigmoid(model([q[b].to(dev) for q in Z])).cpu().numpy())
        return np.concatenate(o)
    best = (-1, None, 0)
    for ep in range(1, 81):
        model.train(); perm = torch.from_numpy(tr)[torch.randperm(len(tr), generator=g)]; thr.reset()
        for i in range(0, len(perm), 32):
            b = perm[i:i + 32]; loss = F.binary_cross_entropy_with_logits(model([q[b].to(dev) for q in Z]), Yt[b].to(dev), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step(); thr.step()
        a = rec_auc(predict(va), va)
        if a > best[0]: best = (a, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
        if ep - best[2] >= 15: break
    model.load_state_dict(best[1]); print(f'   seed {seed}: best epoch {best[2]}, val AUC {best[0]:.3f}, params {sum(p.numel() for p in model.parameters())}', flush=True)
    return predict(te)


def report(P, idx):
    """P: (len(idx), 3 pos, 3 type) probabilities for recordings idx."""
    y, f, d = Y[idx], fc[idx], date[idx]; S = {}
    for c, name in enumerate(TY):
        s = P[:, :, c].max(1); has = y[:, :, c].max(1) > 0; pred = s >= .5
        num = den = 0
        for b in sorted({(dd, ff) for dd, ff in zip(d, f) if ff in (1, 2)}):
            k = (d == b[0]) & (f == b[1])
            if has[k].any() and (~has[k]).any(): a = roc_auc_score(has[k], s[k]); w = has[k].sum() * (~has[k]).sum(); num += a * w; den += w
        one = (y[:, :, c].sum(1) == 1); loc = (P[one][:, :, c].argmax(1) == y[one][:, :, c].argmax(1))
        S[name] = dict(auc=float(roc_auc_score(has, s)), in_block_auc=float(num / den) if den else None, detected=float(pred[has].mean()), n=int(has.sum()),
                       fa_healthy_1109=float(pred[(f == 0) & (d == '11-09')].mean()), fa_healthy_1102=float(pred[(f == 0) & (d == '11-02')].mean()) if ((f == 0) & (d == '11-02')).any() else None,
                       fa_fault_rec_without=float(pred[(f > 0) & ~has].mean()), position_correct=float(loc.mean()), n_position=int(one.sum()))
    pr = P.max(1) >= .5; hs = y.max(1) > 0; anyp = pr.any(1); det = hs.any(1)
    S['any_fault'] = dict(recall=float(anyp[det].mean()), flagged_healthy_1109=float(anyp[(f == 0) & (d == '11-09')].mean()),
                          flagged_healthy_1102=float(anyp[(f == 0) & (d == '11-02')].mean()) if ((f == 0) & (d == '11-02')).any() else None,
                          flagged_only_imb_mod=float(anyp[(f > 0) & ~det].mean()))
    k1 = (f == 1) & det; S['type_exact_1fault'] = float((pr[k1] == hs[k1]).all(1).mean()); S['n_1fault'] = int(k1.sum())
    S['exact_set'] = {f'fc{q}': float((pr[f == q] == hs[f == q]).all(1).mean()) for q in range(4) if (f == q).any()}
    # sector level: is THIS speaker faulty (Whine/PWM/Wind)?
    ss = P.max(2); sy = y.max(2); m = (f == 1) | (f == 2)
    S['sector'] = dict(auc_within_1_2=float(roc_auc_score(sy[m].ravel(), ss[m].ravel())),
                       fc1_fault_sector_ranked_first=float(np.mean([sy[i, ss[i].argmax()] for i in np.where(k1)[0]])),
                       all3_sectors_exact={f'fc{q}': float(((ss[f == q] >= .5) == (sy[f == q] > 0)).all(1).mean()) for q in range(4) if (f == q).any()})
    return S


if __name__ == '__main__':
    dev = torch.device('cuda'); summary = {}
    print('cv5'); P = np.zeros((n, 3, 3), np.float32)
    for k, (tr, te) in enumerate(GroupKFold(5).split(np.arange(n), Y[:, 2, 2], grp)):
        a, b = next(GroupShuffleSplit(1, test_size=.15, random_state=k).split(tr, groups=grp[tr])); P[te] = fit_predict(tr[a], tr[b], te, 100 + k, dev)
    np.save(OUT / 'cv5_probs.npy', P); summary['cv5'] = report(P, np.arange(n))
    print('strict'); uid = np.array([r['uid'] for r in R]); per = []
    for seed in (42, 43, 44):
        sp = read_json(PROJECT / f'runs/strict_split_20261123/seed{seed}.json'); tr, va, te = (np.where(np.isin(uid, sp[k]))[0] for k in ('train', 'validation', 'test'))
        Pt = fit_predict(tr, va, te, seed, dev); np.save(OUT / f'strict_probs_seed{seed}.npy', Pt); per.append(report(Pt, te))
    summary['strict_per_seed'] = per
    write_json(OUT / 'summary.json', summary)
    print(json.dumps(summary['cv5'], indent=1))
    for s in per: print({k: {a: (round(b, 2) if isinstance(b, float) else b) for a, b in v.items()} if isinstance(v, dict) else v for k, v in s.items() if k in ('Whine', 'PWM', 'Wind', 'any_fault')})
