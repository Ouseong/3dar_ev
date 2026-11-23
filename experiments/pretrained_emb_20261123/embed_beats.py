"""BEATs (Microsoft, BEATs_iter3_plus_AS2M, 768-d) embedding per ROI view, mean over frames (as in the penguin notebook).
Code: github.com/microsoft/unilm/beats; checkpoint from the Hugging Face mirror lpepino/beats_ckpts (sha256 d43cbfad...),
loaded with torch.load(weights_only=True). torchaudio has no wheel for torch 2.14.1/aarch64, so only
torchaudio.compliance.kaldi (pure Python, from the torchaudio 2.11.0 wheel) is provided by third_party/torchaudio_shim.
GPU inference only. Output: runs/pretrained_emb_20261123/beats_views.npy (views x 768)
"""
from pathlib import Path
import sys, numpy as np, torch
TP = Path('/home/kos/Desktop/3DAR_EV/third_party'); sys.path.insert(0, str(TP / 'torchaudio_shim')); sys.path.insert(0, str(TP / 'beats'))
from BEATs import BEATs, BEATsConfig
PROJECT = Path(__file__).resolve().parents[2]; out = PROJECT / 'runs/pretrained_emb_20261123'
ck = torch.load(TP / 'beats/BEATs_iter3_plus_AS2M.pt', map_location='cpu', weights_only=True)
m = BEATs(BEATsConfig(ck['cfg'])); m.load_state_dict(ck['model']); m.eval().cuda()
z = np.load(out / 'segments_16k.npz', allow_pickle=True); A = z['audio']; E = np.zeros((len(A), 768), np.float32)
with torch.no_grad():
    for i, a in enumerate(A):
        x = torch.from_numpy(np.asarray(a, np.float32))[None].cuda()
        f, _ = m.extract_features(x, padding_mask=torch.zeros_like(x).bool()); E[i] = f.mean(1)[0].cpu().numpy()
        if i % 1000 == 0: print(i, len(A), flush=True)
np.save(out / 'beats_views.npy', E); print('done', E.shape)
