"""FaultBandNet 3x1 variant (user request 2026-11-24): same inputs and network as the 3x3 model, only the last
layer gives 3 outputs (front / rear / right: faulty or healthy). Label = 1 if that speaker plays Whine/PWM/Wind.
Same repeated source-disjoint splits as train.py (recordings without Imbalance/Modulation).
Output: runs/fault_net_noimbmod_20261124/{probs_3x1.npy, summary_3x1.json}
"""
import sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
PROJECT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PROJECT / 'experiments/fault_net_20261124'))
import train_fault_net as T
from ev3dar.engine import seed_all, GpuThrottle
OUT = PROJECT / 'runs/fault_net_noimbmod_20261124'; FO = json.load(open(OUT / 'folds.json'))
Y1 = T.Y.max(2)                                                              # (n, 3) sector faulty


class FaultBandNet3x1(T.FaultBandNet):
    def __init__(s):
        super().__init__(); s.out = nn.Linear(64 * 36, 3)
    def forward(s, xs):
        h = torch.cat([b(x) for b, x in zip(s.b, xs)], 1); h = F.gelu(s.a2(s.drop(F.gelu(s.a1(h)))))
        return s.out(s.drop(h.flatten(1)))


def fit_predict(tr, va, te, seed, dev):
    seed_all(seed); mu = [x[tr].mean() for x in T.X]; sd = [x[tr].std() for x in T.X]; Z = [(x - m) / s for x, m, s in zip(T.X, mu, sd)]
    Yt = torch.from_numpy(Y1); pos = Y1[tr].sum(0); pw = torch.tensor(np.clip((len(tr) - pos) / np.maximum(pos, 1), 1, 20), dtype=torch.float32, device=dev)
    model = FaultBandNet3x1().to(dev); opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-3); g = torch.Generator().manual_seed(seed); thr = GpuThrottle()
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
        a = roc_auc_score(Y1[va].ravel(), predict(va).ravel())
        if a > best[0]: best = (a, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
        if ep - best[2] >= 15: break
    model.load_state_dict(best[1]); print(f'   rep seed {seed}: best epoch {best[2]}, val sector AUC {best[0]:.3f}', flush=True)
    return predict(te)


if __name__ == '__main__':
    ix = {r['uid']: i for i, r in enumerate(T.R)}; dev = torch.device('cuda')
    S = np.zeros((T.n, 3)); C = np.zeros(T.n)
    for k, rep in enumerate(FO['reps']):
        tp = np.array([ix[u] for u in rep['train_pool']]); te = np.array([ix[u] for u in rep['test']])
        a, b = next(GroupShuffleSplit(1, test_size=.15, random_state=k).split(tp, groups=T.grp[tp]))
        S[te] += fit_predict(tp[a], tp[b], te, 1000 + k, dev); C[te] += 1
    idx = np.where(C > 0)[0]; P = (S[idx] / C[idx, None]).astype(np.float32); np.save(OUT / 'probs_3x1.npy', P)
    y = Y1[idx] > 0; f = T.fc[idx]; d = T.date[idx]; pred = P >= .5; summ = dict(n_tested=int(len(idx)))
    summ['sector_auc_all'] = float(roc_auc_score(y.ravel(), P.ravel()))
    m12 = (f == 1) | (f == 2); summ['sector_auc_fc1_2'] = float(roc_auc_score(y[m12].ravel(), P[m12].ravel()))
    for q in range(4):
        m = f == q; H = ~y[m]; Fm = y[m]; p = pred[m]
        summ[f'fc{q}'] = dict(n_rec=int(m.sum()), healthy_sectors=int(H.sum()), healthy_correct=int((~p[H]).sum()),
                              fault_sectors=int(Fm.sum()), fault_correct=int(p[Fm].sum()), sector_acc=float((p == Fm).mean()),
                              all3_correct=float((p == Fm).all(1).mean()))
    k1 = f == 1; summ['fc1_fault_sector_ranked_first'] = float(np.mean([y[i, P[i].argmax()] for i in np.where(k1)[0]]))
    summ['healthy_rec_flagged'] = {dd: float(pred[(f == 0) & (d == dd)].any(1).mean()) for dd in sorted(set(d[f == 0]))}
    json.dump(summ, open(OUT / 'summary_3x1.json', 'w'), indent=1); print(json.dumps(summ, indent=1))
