"""Supervised and unsupervised Healthy/Fault on the ORIGINAL KU Leuven files only (no 3DAR recording). 2026-11-23.
Data: 2200 files (Healthy 1200, 5 fault classes x 200). Segment = the 5000-8000 RPM part of each run-up (RPM curve from
analyze_originals.py), i.e. the range played in the 3DAR experiment. Two inputs, time axis resampled to 256 frames:
  mel  : 64-band log-mel, 0-12 kHz (same config as the recorded-data experiments)
  stft : linear STFT 0-6 kHz, n_fft 4096 (11.7 Hz bins) -> 512 rows
Split: file level 8:1:1 stratified by class, seeds 42/43/44. Per-frequency z-score from the training files.
SUPERVISED: CNN of the senior-memo structure (Conv 32/64/128 + GroupNorm + GELU + MaxPool -> avg pool -> FCN 128-64 ->
  2 classes), class-weighted cross-entropy, AdamW 1e-4, <= 40 epochs, best validation AUC.
UNSUPERVISED: convolutional autoencoder trained on Healthy training files only (stride-2 convs -> latent 64 -> mirror),
  MSE, AdamW 1e-3, <= 60 epochs, early stop on Healthy validation reconstruction. Score = reconstruction MSE.
  Evaluated on test Healthy vs ALL fault files (no fault file is ever used for the AE).
GPU duty ~50 % (GpuThrottle). Output: runs/ku_original_20261123/train_originals_summary.json
"""
from pathlib import Path
import sys, json, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np, soundfile as sf, torch
from torch import nn
import torch.nn.functional as F
from scipy import signal
from sklearn.metrics import roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(Path(__file__).parent))
from ev3dar.common import load_config, read_json, write_json
from ev3dar.features import logmel
from ev3dar.engine import seed_all, GpuThrottle
import analyze_originals as A
OUT = PROJECT / 'runs/ku_original_20261123'; CFG = load_config(PROJECT / 'runs/baseline/config.resolved.json'); NT = 256
CL = A.CLASSES


def rs(x):  # resample time axis to NT frames
    src = np.linspace(0, x.shape[1] - 1, NT); return np.stack([np.interp(src, np.arange(x.shape[1]), r) for r in x])


def feat(f):
    a, fs = sf.read(A.KU / f['cls'] / f['name'], dtype='float32'); fn = A.rpm_model(f['c'], len(a) / fs)
    r = fn(np.arange(len(a)) / fs); seg = a[int(np.searchsorted(r, 5000)):int(np.searchsorted(r, 8000))]
    mel, _ = logmel(seg, CFG)
    fr, _, p = signal.spectrogram(seg.astype(np.float64), fs=fs, window='hann', nperseg=4096, noverlap=4096 - 480, detrend=False, scaling='spectrum', mode='psd')
    st = 10 * np.log10(np.maximum(p[:512], 1e-12))
    return rs(mel).astype(np.float16), rs(st).astype(np.float16)


class CNN(nn.Module):
    def __init__(s):
        super().__init__(); L = []; c = 1
        for o in (32, 64, 128): L += [nn.Conv2d(c, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(), nn.MaxPool2d(2)]; c = o
        s.enc = nn.Sequential(*L); s.fcn = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Dropout(.2), nn.Linear(128, 64), nn.GELU()); s.dec = nn.Linear(64, 2)
    def forward(s, x): return s.dec(s.fcn(s.enc(x).mean((2, 3))))


class AE(nn.Module):
    def __init__(s, H, latent=64):
        super().__init__(); n = 3 if H == 64 else 5; ch = [1, 16, 32, 64, 128, 128][:n + 1]
        s.enc = nn.Sequential(*[m for i in range(n) for m in (nn.Conv2d(ch[i], ch[i + 1], 3, 2, 1), nn.BatchNorm2d(ch[i + 1]), nn.ReLU())])
        s.shape = (ch[n], H >> n, NT >> n); d = int(np.prod(s.shape)); s.to_z = nn.Linear(d, latent); s.from_z = nn.Linear(latent, d)
        dec = []
        for i in range(n, 0, -1):
            dec += [nn.ConvTranspose2d(ch[i], ch[i - 1], 4, 2, 1)] + ([nn.BatchNorm2d(ch[i - 1]), nn.ReLU()] if i > 1 else [])
        s.dec = nn.Sequential(*dec)
    def forward(s, x): return s.dec(s.from_z(s.to_z(s.enc(x).flatten(1))).view(-1, *s.shape))


def batches(n, bs, g=None):
    idx = torch.randperm(n, generator=g) if g is not None else torch.arange(n)
    for i in range(0, n, bs): yield idx[i:i + bs]


def run(X, y, cls, seed, dev):
    rng = np.random.default_rng(seed); tr, va, te = [], [], []
    for c in CL:
        i = rng.permutation(np.where(cls == c)[0]); a, b = int(.8 * len(i)), int(.9 * len(i)); tr += list(i[:a]); va += list(i[a:b]); te += list(i[b:])
    tr, va, te = map(np.array, (tr, va, te)); res = {}
    # ---------------- supervised ----------------
    seed_all(seed); m, s = X[tr].mean((0, 2), keepdims=True), X[tr].std((0, 2), keepdims=True) + 1e-3
    Z = torch.from_numpy(((X - m) / s).astype(np.float32))[:, None]; Y = torch.from_numpy(y)
    model = CNN().to(dev); opt = torch.optim.AdamW(model.parameters(), 1e-4, weight_decay=1e-5); thr = GpuThrottle(); g = torch.Generator().manual_seed(seed)
    w = torch.tensor([len(tr) / (2 * (y[tr] == k).sum()) for k in (0, 1)], dtype=torch.float32, device=dev)
    def prob(ix):
        model.eval(); o = []
        with torch.no_grad():
            for b in batches(len(ix), 32): o.append(model(Z[ix[b]].to(dev)).softmax(-1)[:, 1].cpu().numpy())
        return np.concatenate(o)
    best = (-1, None, 0)
    for ep in range(1, 41):
        model.train(); thr.reset()
        for b in batches(len(tr), 16, g):
            loss = F.cross_entropy(model(Z[tr[b]].to(dev)), Y[tr[b]].to(dev), weight=w); opt.zero_grad(); loss.backward(); opt.step(); thr.step()
        a = roc_auc_score(y[va], prob(va))
        if a > best[0]: best = (a, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
        if ep - best[2] >= 10: break
    model.load_state_dict(best[1]); p = prob(te); pred = p >= .5
    res['supervised'] = dict(auc=float(roc_auc_score(y[te], p)), accuracy=float((pred == y[te]).mean()), best_epoch=best[2],
                             specificity=float(1 - pred[y[te] == 0].mean()), **{'recall_' + c: float(pred[cls[te] == c].mean()) for c in CL[1:]})
    # ---------------- unsupervised AE ----------------
    seed_all(seed); h_tr, h_va, h_te = tr[y[tr] == 0], va[y[va] == 0], te[y[te] == 0]
    m, s = X[h_tr].mean((0, 2), keepdims=True), X[h_tr].std((0, 2), keepdims=True) + 1e-3
    Z = torch.from_numpy(((X - m) / s).astype(np.float32))[:, None]
    ae = AE(X.shape[1]).to(dev); opt = torch.optim.AdamW(ae.parameters(), 1e-3, weight_decay=1e-5); g = torch.Generator().manual_seed(seed)
    def err(ix):
        ae.eval(); o = []
        with torch.no_grad():
            for b in batches(len(ix), 32): x = Z[ix[b]].to(dev); o.append(((ae(x) - x) ** 2).mean((1, 2, 3)).cpu().numpy())
        return np.concatenate(o)
    best = (1e9, None, 0)
    for ep in range(1, 61):
        ae.train(); thr.reset()
        for b in batches(len(h_tr), 16, g):
            x = Z[h_tr[b]].to(dev); loss = ((ae(x) - x) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step(); thr.step()
        v = float(err(h_va).mean())
        if v < best[0]: best = (v, {k: v_.detach().cpu().clone() for k, v_ in ae.state_dict().items()}, ep)
        if ep - best[2] >= 10: break
    ae.load_state_dict(best[1]); eh = err(h_te); allf = np.where(y == 1)[0]; ef = err(allf)
    res['unsupervised_ae'] = dict(auc_all_faults=float(roc_auc_score(np.r_[np.zeros(len(eh)), np.ones(len(ef))], np.r_[eh, ef])), best_epoch=best[2],
                                  val_healthy_mse=best[0],
                                  **{'auc_' + c: float(roc_auc_score(np.r_[np.zeros(len(eh)), np.ones((cls[allf] == c).sum())], np.r_[eh, ef[cls[allf] == c]])) for c in CL[1:]})
    return res


if __name__ == '__main__':
    FF = read_json(OUT / 'file_features.json'); cache = OUT / 'orig_inputs.npz'
    if cache.exists(): z = np.load(cache); MEL, ST = z['mel'], z['stft']
    else:
        with ProcessPoolExecutor(12) as ex: r = list(ex.map(feat, FF, chunksize=8))
        MEL, ST = np.stack([a for a, _ in r]), np.stack([b for _, b in r]); np.savez(cache, mel=MEL, stft=ST)
    cls = np.array([f['cls'] for f in FF]); y = (cls != 'Healthy').astype(np.int64); dev = torch.device('cuda')
    S = {}
    for name, X in [('mel64', MEL), ('stft512', ST)]:
        X = np.clip(X.astype(np.float32), -120, 0); runs = [run(X, y, cls, sd, dev) for sd in (42, 43, 44)]
        for part in ('supervised', 'unsupervised_ae'):
            S[f'{name} | {part}'] = {k: dict(seeds=[round(r[part][k], 3) for r in runs], mean=round(float(np.mean([r[part][k] for r in runs])), 3)) for k in runs[0][part]}
        write_json(OUT / 'train_originals_summary.json', S)
        for part in ('supervised', 'unsupervised_ae'): print('==', name, part, {k: v['mean'] for k, v in S[f'{name} | {part}'].items()}, flush=True)
