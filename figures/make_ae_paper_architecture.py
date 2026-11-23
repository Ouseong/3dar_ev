"""Schematic of the paper autoencoder (Kunte et al., Fig. 10) applied to the EV per-ROI data
(experiments/ae_paper_20261123/train_ae_paper.py, n=6 blocks, f=8 filters) -> SVG.
Same visual style as make_ae_architecture.py. Usage: python make_ae_paper_architecture.py figures/ae_paper_architecture
"""
import sys, math
from pathlib import Path

OUT = Path(sys.argv[1])
C = dict(conv='#C9E6F7', norm='#F3F5C4', attn='#FCE1C6', emb='#F8D3D7', lin='#DCDDF3', soft='#CFEBD3',
         box='#F3F3F3', loss='#FFFFFF', pool='#E9D8F2', up='#D6EFE6')
S = []
N, F = 6, 8


def esc(s): return s.replace('&', '&amp;').replace('<', '&lt;')


def text(cx, cy, lines, fs=12, bold_first=False, anchor='middle', color='#222', italic=False):
    if isinstance(lines, str): lines = [lines]
    lh = fs * 1.22; y0 = cy - (len(lines) - 1) * lh / 2 + fs * 0.36
    for i, l in enumerate(lines):
        fw = '600' if (bold_first and i == 0) else '400'
        sz = fs if (i == 0 or not bold_first) else fs - 1.5
        S.append(f'<text x="{cx}" y="{y0 + i * lh:.1f}" font-size="{sz}" font-weight="{fw}" text-anchor="{anchor}" '
                 f'fill="{color}"{" font-style=\"italic\"" if italic else ""}>{esc(l)}</text>')


def rect(x, y, w, h, fill, lines, fs=12, bold_first=False, rx=6, dash=False):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="#555" '
             f'stroke-width="1.1"{" stroke-dasharray=\"4 3\"" if dash else ""}/>')
    text(x + w / 2, y + h / 2, lines, fs, bold_first)


def arrow(pts, dash=False):
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    L = math.hypot(x2 - x1, y2 - y1) or 1; ux, uy = (x2 - x1) / L, (y2 - y1) / L
    bx_, by_ = x2 - ux * 9, y2 - uy * 9
    d = 'M' + ' L'.join(f'{x},{y}' for x, y in pts[:-1] + [(bx_, by_)])
    S.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="1.4"' + (' stroke-dasharray="4 3"' if dash else '') + '/>')
    S.append(f'<polygon points="{x2},{y2} {bx_ - uy * 4.5:.1f},{by_ + ux * 4.5:.1f} {bx_ + uy * 4.5:.1f},{by_ - ux * 4.5:.1f}" fill="#222"/>')


def line(pts):
    S.append('<path d="M' + ' L'.join(f'{x},{y}' for x, y in pts) + '" fill="none" stroke="#222" stroke-width="1.4"/>')


def container(x, y0, y1, w, label, note, at):
    S.insert(at, f'<rect x="{x - w / 2}" y="{y0}" width="{w}" height="{y1 - y0}" rx="12" fill="{C["box"]}" '
             f'stroke="#888" stroke-width="1.2" stroke-dasharray="6 4"/>')
    text(x - w / 2 - 14, (y0 + y1) / 2 - 10, label, 13, anchor='end')
    text(x - w / 2 - 14, (y0 + y1) / 2 + 10, note, 11, anchor='end', color='#666', italic=True)


W, H = 1240, 1560


def polar(cx, cy, r, deg):
    a = math.radians(deg); return cx + r * math.cos(a), cy - r * math.sin(a)


def arc(cx, cy, r, a0, a1, fill):
    x0, y0 = polar(cx, cy, r, a0); x1, y1 = polar(cx, cy, r, a1); large = 1 if (a1 - a0) % 360 > 180 else 0
    S.append(f'<path d="M{cx},{cy} L{x0:.1f},{y0:.1f} A{r},{r} 0 {large} 0 {x1:.1f},{y1:.1f} Z" fill="{fill}" stroke="#555" stroke-width="1"/>')


# ================= (a) Input pipeline (same as 11/19 AE except scaling / padding) =================
ax = 290; dy = H - 1320
text(ax, 34, '(a) Input pipeline: 3 angular ROIs', 17, True)
text(ax, 56, 'runs/baseline cache (id_views) · same ROIs / splits as 11/19 AE', 11, color='#666', italic=True)
text(ax, 360 + dy, ['Unit = one ROI of one recording (1,200 rec × 3 ROIs = 3,600)', 'Healthy 1,800 · Fault 1,800 (5 types)'], 12, color='#444')
text(ax, 1222 + dy, ['Inputs', '3-speaker recording (~4 s, 48 kHz) + encoder angle CSV'], 13, True)
arrow([(ax, 1196 + dy), (ax, 1158 + dy)])
rect(ax - 140, 1110 + dy, 280, 48, C['emb'], ['Log-Mel Spectrogram', '64 mel · n_fft 2048 · hop 480 · 0–12 kHz · dBFS'], 13, True)
arrow([(ax, 1110 + dy), (ax, 1078 + dy)])
rect(ax - 140, 1030 + dy, 280, 48, C['emb'], ['Angle → Time (encoder)', 'each ROI = the time span of its angles'], 13, True)
arrow([(ax, 1030 + dy), (ax, 1000 + dy)])
pcx, pcy, pr = ax, 920 + dy, 70
arc(pcx, pcy, pr, 135, 225, C['attn']); arc(pcx, pcy, pr, 225, 315, C['soft']); arc(pcx, pcy, pr, 315, 405, C['lin'])
arc(pcx, pcy, pr, 45, 135, '#FFFFFF')
S.append(f'<circle cx="{pcx}" cy="{pcy}" r="22" fill="white" stroke="#555"/>')
text(pcx, pcy, 'mic', 10, color='#444')
for d, lab, anc in [(0, '0°', 'start'), (90, '90°', 'start'), (180, '180°', 'end'), (270, '270°', 'start')]:
    x, y = polar(pcx, pcy, pr + 12, d); x += 8 if d in (90, 270) else 0
    text(x, y + (4 if d in (0, 180) else (-2 if d == 90 else 8)), lab, 11, anchor=anc, color='#444')
for d, lab in [(180, 'F'), (270, 'R'), (0, 'B')]:
    x, y = polar(pcx, pcy, 48, d); text(x, y, lab, 12, True)
text(pcx + 150, pcy, ['unused 45–135°', '(no ROI)'], 11, color='#666', italic=True)
fy = 812 + dy; xs = [ax - 175, ax, ax + 175]
line([(ax, pcy - pr - 4), (ax, fy)]); line([(xs[0], fy), (xs[2], fy)])
for x in xs: arrow([(x, fy), (x, 786 + dy)])
rect(xs[0] - 80, 728 + dy, 160, 58, C['attn'], ['FRONT 135–225°', '1 view · 64 × ~100'], 12, True)
rect(xs[1] - 80, 728 + dy, 160, 58, C['soft'], ['RIGHT 225–315°', '1 view · 64 × ~100'], 12, True)
rect(xs[2] - 80, 728 + dy, 160, 58, C['lin'], ['REAR 315–360 | 0–45°', '2 views, NOT spliced'], 12, True)
for x in xs: line([(x, 728 + dy), (x, 706 + dy)])
line([(xs[0], 706 + dy), (xs[2], 706 + dy)]); arrow([(ax, 706 + dy), (ax, 676 + dy)])
rect(ax - 165, 618 + dy, 330, 58, C['norm'], ['Paper scaling: clip −120…0 dBFS → [0, 1]',
     'dataset-wide (not per sample) · no z-score', 'pad 100 → 128 frames · padded frames masked'], 12, True)
arrow([(ax, 618 + dy), (ax, 586 + dy)])
rect(ax - 190, 480 + dy, 380, 106, C['box'], ['Recording-level split (seed 42 / 43 / 44)', '8 : 1 : 1, stratified fault-count × RPM band',
     'train: Healthy ROIs only (1,440)', 'validation: Healthy + Fault (360) → stop & threshold', 'test: Healthy + Fault (360) → final score'], 12, True)
arrow([(ax, 480 + dy), (ax, 442 + dy)])
text(ax, 422 + dy, ['to autoencoder (b)', 'one view at a time: 1 × 64 × 128'], 13, True)

# ================= (b) Paper CNN autoencoder, n = 6, f = 8 =================
bx = 860; bw = 300
text(bx, 34, f'(b) Paper CNN autoencoder (Kunte et al., Fig. 10) · n = {N}, f = {F}', 17, True)
text(bx, 56, 'experiments/ae_paper_20261123/train_ae_paper.py · 7,753 parameters · no dense layer', 11, color='#666', italic=True)

y = H - 50
text(bx, y, ['Input view x', '1 × 64 mel × 128 frames'], 13, True)
y -= 26; arrow([(bx, y), (bx, y - 24)]); y -= 24
# encoder blocks, bottom-up
h_ = 64; w_ = 128
enc_top = y; enc_bot = y; enc_at = len(S)
y -= 10
for i in range(N):
    cin = 1 if i == 0 else F
    hh, ww = h_ // 2 ** (i + 1), w_ // 2 ** (i + 1)
    y -= 32; rect(bx - 130, y, 260, 32, C['conv'], [f'Conv2d 3×3 s1 p1  {cin} → {F}', f'{h_ // 2 ** i} × {w_ // 2 ** i}'], 11.5, True)
    y -= 22; rect(bx - 130, y, 260, 22, C['norm'], 'BatchNorm2d · ReLU', 11)
    y -= 30; rect(bx - 130, y, 260, 30, C['pool'], f'MaxPool2d 2×2  →  {F} × {hh} × {ww}', 11.5)
    if i < N - 1: arrow([(bx, y), (bx, y - 10)]); y -= 10
y -= 10; enc_top = y
container(bx, enc_top, enc_bot, bw, 'Encoder', f'block × n = {N}', enc_at)
arrow([(bx, y), (bx, y - 24)]); y -= 24
y -= 44; rect(bx - 110, y, 220, 44, C['attn'], ['Encoded representation z', f'{F} × 1 × 2  (= 16 values)'], 12, True)
arrow([(bx, y), (bx, y - 24)]); y -= 24
# decoder blocks
dec_bot = y; y -= 10; dec_at = len(S)
for i in range(N - 1):
    hh, ww = 2 ** (i + 1), 2 ** (i + 2)
    y -= 30; rect(bx - 130, y, 260, 30, C['up'], f'ConvTranspose2d k2 s2  {F} → {F}  →  {hh} × {ww}', 11.5)
    y -= 32; rect(bx - 130, y, 260, 32, C['conv'], [f'Conv2d 3×3 s1 p1  {F} → {F}', 'BatchNorm2d · ReLU'], 11.5, True)
    if i < N - 2: arrow([(bx, y), (bx, y - 10)]); y -= 10
y -= 10; dec_top = y
container(bx, dec_top, dec_bot, bw, 'Decoder', f'block × (n−1) = {N - 1}', dec_at)
arrow([(bx, y), (bx, y - 22)]); y -= 22
out_bot = y
y -= 30; rect(bx - 130, y, 260, 30, C['up'], f'ConvTranspose2d k2 s2  {F} → {F}  →  64 × 128', 11.5)
y -= 30; rect(bx - 130, y, 260, 30, C['conv'], f'Conv2d 3×3 s1 p1  {F} → 1', 11.5, True)
y -= 22; rect(bx - 130, y, 260, 22, C['soft'], 'Sigmoid  (output in [0, 1])', 11)
text(bx - bw / 2 - 14, (y + out_bot) / 2, 'Output', 13, anchor='end')
arrow([(bx, y), (bx, y - 22)]); y -= 22
text(bx, y - 10, 'Reconstruction x_rec  1 × 64 × 128', 13, True); rec_y = y - 10
y -= 30; arrow([(bx, y), (bx, y - 18)]); y -= 18
y -= 50; rect(bx - 150, y, 300, 50, C['emb'], ['Anomaly score per ROI', 'masked MSE(x, x_rec) over valid frames',
                                               'REAR: duration-weighted mean of 2 views'], 11.5, True)
arrow([(bx, y), (bx, y - 18)]); y -= 18
y -= 40; rect(bx - 150, y, 300, 40, C['attn'], ['Threshold from validation', 'maximize balanced accuracy'], 11.5, True)
arrow([(bx, y), (bx, y - 18)]); y -= 18
text(bx, y - 18, ['Healthy / Fault per ROI', 'AUC · accuracy · recall · per position / type'], 13, True)

# training box (paper Table 5) on the right
lx = bx + bw / 2 + 25; ly = 330
rect(lx, ly, 150, 200, C['loss'], ['Training (Table 5)', 'Healthy views only', 'loss: masked MSE',
     'Adam · lr 1e-4', 'batch 8 · ≤ 100 epochs', 'early stop: val', 'Healthy recon. (pat. 15)', 'GPU ~50%'], 11.5, True, dash=True)
arrow([(bx + 130, rec_y), (lx + 75, rec_y), (lx + 75, ly)], dash=True)
rect(lx, 600, 150, 190, C['loss'], ['Our choices', '(not in paper)', 'ConvT k2 s2 to undo', 'each MaxPool', 'last Conv → 1 channel',
     'dBFS: clip −120…0', 'time pad → 128', 'n, f fixed to paper pick'], 11.5, True, dash=True)

S.append(f'<line x1="{W / 2 - 10}" y1="80" x2="{W / 2 - 10}" y2="{H - 60}" stroke="#ccc" stroke-dasharray="5 5"/>')
legend = [(C['emb'], 'input / score'), (C['conv'], 'convolution'), (C['norm'], 'normalization / scaling'),
          (C['pool'], 'max-pool'), (C['up'], 'transposed conv'), (C['attn'], 'bottleneck / decision'), (C['box'], 'repeated block')]
lgx = 60
for col, lab in legend:
    S.append(f'<rect x="{lgx}" y="{H - 22}" width="14" height="14" rx="3" fill="{col}" stroke="#555"/>')
    S.append(f'<text x="{lgx + 20}" y="{H - 11}" font-size="11.5" fill="#333">{lab}</text>')
    lgx += 20 + len(lab) * 6.0 + 22
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Helvetica, Arial, \'DejaVu Sans\', sans-serif">'
       f'<rect width="{W}" height="{H}" fill="white"/>' + ''.join(S) + '</svg>')
OUT.with_suffix('.svg').write_text(svg, encoding='utf-8')
print('ok', OUT, 'top of (b) at y =', round(y))
