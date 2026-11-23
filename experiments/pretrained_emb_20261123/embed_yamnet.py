"""YAMNet (Google, AudioSet, 1024-d) embedding per ROI view, mean over frames; views shorter than 0.975 s are zero-padded
to 0.975 s (as in the penguin notebook). Official code tensorflow/models research/audioset/yamnet + yamnet.h5
(storage.googleapis.com/audioset). Runs in the separate venv /home/kos/Desktop/3DAR_EV/.venv_tf (TensorFlow 2.18, CPU).
Output: runs/pretrained_emb_20261123/yamnet_views.npy (views x 1024)
"""
from pathlib import Path
import sys, numpy as np
TP = Path('/home/kos/Desktop/3DAR_EV/third_party/yamnet'); sys.path.insert(0, str(TP))
import params as P, yamnet as Y
out = Path(__file__).resolve().parents[2] / 'runs/pretrained_emb_20261123'
m = Y.yamnet_frames_model(P.Params()); m.load_weights(str(TP / 'yamnet.h5'))
z = np.load(out / 'segments_16k.npz', allow_pickle=True); A = z['audio']; E = np.zeros((len(A), 1024), np.float32)
minlen = int(0.975 * 16000)
for i, a in enumerate(A):
    a = np.asarray(a, np.float32)
    if len(a) < minlen: a = np.pad(a, (0, minlen - len(a)))
    _, e, _ = m(a); E[i] = e.numpy().mean(0)
    if i % 1000 == 0: print(i, len(A), flush=True)
np.save(out / 'yamnet_views.npy', E); print('done', E.shape)
