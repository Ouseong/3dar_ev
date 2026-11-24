"""Same-style schematic of the normal-only ROI autoencoder (2026-11-19) -> SVG + PNG."""
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
C = dict(conv='#C9E6F7', norm='#F3F5C4', attn='#FCE1C6', emb='#F8D3D7', lin='#DCDDF3', soft='#CFEBD3',
         frozen='#E4E4E4', box='#F3F3F3', stroke='#555', text='#222', loss='#FFFFFF')
S = []

def rect(x, y, w, h, fill, lines, fs=12, bold_first=False, rx=6, dash=False, stroke='#555'):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" '
             f'stroke-width="1.1"{" stroke-dasharray=\"4 3\"" if dash else ""}/>')
    text(x + w / 2, y + h / 2, lines, fs, bold_first)

def text(cx, cy, lines, fs=12, bold_first=False, anchor='middle', color=None, italic=False):
    if isinstance(lines, str): lines = [lines]
    lh = fs * 1.22; y0 = cy - (len(lines) - 1) * lh / 2 + fs * 0.36
    for i, l in enumerate(lines):
        l = l.replace('&', '&amp;').replace('<', '&lt;')
        fw = '600' if (bold_first and i == 0) else '400'
        sz = fs if (i == 0 or not bold_first) else fs - 1.5
        S.append(f'<text x="{cx}" y="{y0 + i * lh:.1f}" font-size="{sz}" font-weight="{fw}" text-anchor="{anchor}" '
                 f'fill="{color or C["text"]}"{" font-style=\"italic\"" if italic else ""}>{l}</text>')

def arrow(pts, dash=False):
    import math
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    L = math.hypot(x2 - x1, y2 - y1) or 1; ux, uy = (x2 - x1) / L, (y2 - y1) / L
    bx_, by_ = x2 - ux * 9, y2 - uy * 9
    d = 'M' + ' L'.join(f'{x},{y}' for x, y in pts[:-1] + [(bx_, by_)])
    S.append(f'<path d="{d}" fill="none" stroke="#222" stroke-width="1.4"' +
             (' stroke-dasharray="4 3"' if dash else '') + '/>')
    S.append(f'<polygon points="{x2},{y2} {bx_ - uy * 4.5:.1f},{by_ + ux * 4.5:.1f} {bx_ + uy * 4.5:.1f},{by_ - ux * 4.5:.1f}" fill="#222"/>')

def line(pts):
    S.append('<path d="M' + ' L'.join(f'{x},{y}' for x, y in pts) + '" fill="none" stroke="#222" stroke-width="1.4"/>')

import math
W, H = 1120, 1320
def polar(cx, cy, r, deg):  # 0 deg at right, counter-clockwise (schematic only)
    a = math.radians(deg); return cx + r * math.cos(a), cy - r * math.sin(a)
def arc(cx, cy, r, a0, a1, fill):
    x0, y0 = polar(cx, cy, r, a0); x1, y1 = polar(cx, cy, r, a1); large = 1 if (a1 - a0) % 360 > 180 else 0
    S.append(f'<path d="M{cx},{cy} L{x0:.1f},{y0:.1f} A{r},{r} 0 {large} 0 {x1:.1f},{y1:.1f} Z" fill="{fill}" stroke="#555" stroke-width="1"/>')

# ================= (a) Data: recording -> 3 angular ROIs =================
ax = 290
text(ax, 34, '(a) Input pipeline: 3 angular ROIs', 17, True)
text(ax, 56, 'runs/baseline cache (id_views) · experiments/ae_20261119/train_ae.py', 11, color='#666', italic=True)
text(ax, 140, ['Unit = one ROI of one recording (1,200 rec × 3 ROIs = 3,600)', 'Healthy 1,800 · Fault 1,800 (5 types)'], 12, color='#444')

text(ax, 1222, ['Inputs', '3-speaker recording (~4 s, 48 kHz) + encoder angle CSV'], 13, True)
arrow([(ax, 1196), (ax, 1158)])
rect(ax - 140, 1110, 280, 48, C['emb'], ['Log-Mel Spectrogram', '64 mel · n_fft 2048 · hop 480 · 0–12 kHz · dB'], 13, True)
arrow([(ax, 1110), (ax, 1078)])
rect(ax - 140, 1030, 280, 48, C['emb'], ['Angle → Time (encoder)', 'each ROI = the time span of its angles'], 13, True)
arrow([(ax, 1030), (ax, 1000)])
# polar ROI ring
pcx, pcy, pr = ax, 920, 70
arc(pcx, pcy, pr, 135, 225, C['attn']); arc(pcx, pcy, pr, 225, 315, C['soft']); arc(pcx, pcy, pr, 315, 405, C['lin'])
arc(pcx, pcy, pr, 45, 135, '#FFFFFF')
S.append(f'<circle cx="{pcx}" cy="{pcy}" r="22" fill="white" stroke="#555"/>')
text(pcx, pcy, 'mic', 10, color='#444')
for d, lab, anc in [(0, '0°', 'start'), (90, '90°', 'start'), (180, '180°', 'end'), (270, '270°', 'start')]:
    x, y = polar(pcx, pcy, pr + 12, d); x += 8 if d in (90, 270) else 0; text(x, y + (4 if d in (0, 180) else (-2 if d == 90 else 8)), lab, 11, anchor=anc, color='#444')
x, y = polar(pcx, pcy, 48, 180); text(x, y, 'F', 12, True)
x, y = polar(pcx, pcy, 48, 270); text(x, y, 'R', 12, True)
x, y = polar(pcx, pcy, 48, 0); text(x, y, 'B', 12, True)
x, y = polar(pcx, pcy, 48, 90); text(x, y, '—', 11, color='#888')
text(pcx + 150, pcy, ['unused 45–135°', '(no ROI)'], 11, color='#666', italic=True)
# fork into three ROI views
fy = 812; xs = [ax - 175, ax, ax + 175]
line([(ax, pcy - pr - 4), (ax, fy)]); line([(xs[0], fy), (xs[2], fy)])
for x in xs: arrow([(x, fy), (x, 786)])
rect(xs[0] - 80, 728, 160, 58, C['attn'], ['FRONT 135–225°', '1 view · 64 × ~100'], 12, True)
rect(xs[1] - 80, 728, 160, 58, C['soft'], ['RIGHT 225–315°', '1 view · 64 × ~100'], 12, True)
rect(xs[2] - 80, 728, 160, 58, C['lin'], ['REAR 315–360 | 0–45°', '2 views, NOT spliced'], 12, True)
for x in xs: line([(x, 728), (x, 706)])
line([(xs[0], 706), (xs[2], 706)]); arrow([(ax, 706), (ax, 676)])
rect(ax - 150, 628, 300, 48, C['norm'], ['Per-mel z-score + pad 100 → 104 frames', 'mean / std from train Healthy only · mask'], 12, True)
arrow([(ax, 628), (ax, 596)])
rect(ax - 190, 490, 380, 106, C['box'], ['Recording-level split (seed 42 / 43 / 44)', '8 : 1 : 1, stratified fault-count × RPM band',
     'train: Healthy ROIs only (1,440)', 'validation: Healthy + Fault (360) → stop & threshold', 'test: Healthy + Fault (360) → final score'], 12, True)
arrow([(ax, 490), (ax, 452)])
text(ax, 432, ['to autoencoder (b)', 'one view at a time: 1 × 64 × 104'], 13, True)

# ================= (b) Convolutional autoencoder =================
bx = 830
text(bx, 34, '(b) Normal-only convolutional autoencoder', 17, True)
text(bx, 56, 'ConvAE · latent 64 · fault = large reconstruction error', 11, color='#666', italic=True)
text(bx, 140, ['learns to rebuild Healthy spectrograms only', 'fault labels: threshold (validation) and scoring (test) only'], 12, color='#444')

text(bx, 1240, ['Input view x', '1 × 64 mel × 104 frames'], 13, True)
arrow([(bx, 1214), (bx, 1188)])
e0, e1 = 990, 1188
S.append(f'<rect x="{bx - 150}" y="{e0}" width="300" height="{e1 - e0}" rx="12" fill="{C["box"]}" stroke="#888" stroke-width="1.2"/>')
text(bx - 190, (e0 + e1) / 2, ['Encoder'], 13)
enc = [('Conv2d k3 s2  1 → 32', '64×104 → 32×52'), ('Conv2d k3 s2  32 → 64', '→ 16×26'), ('Conv2d k3 s2  64 → 128', '→ 8×13')]
y = e1 - 10
for i, (a, b) in enumerate(enc):
    y -= 62
    rect(bx - 120, y + 34, 240, 20, C['norm'], 'BatchNorm · ReLU', 11)
    rect(bx - 120, y, 240, 34, C['conv'], [a, b], 12, True)
    if i < 2: arrow([(bx, y), (bx, y - 8)])
line([(bx, y), (bx, e0)])
arrow([(bx, e0), (bx, 966)])
rect(bx - 120, 930, 240, 36, C['lin'], ['Flatten · Linear 13,312 → 64'], 12)
arrow([(bx, 930), (bx, 908)])
rect(bx - 90, 870, 180, 38, C['attn'], ['Latent z (64)', 'bottleneck'], 12, True)
arrow([(bx, 870), (bx, 848)])
rect(bx - 120, 812, 240, 36, C['lin'], ['Linear 64 → 13,312 · reshape 128×8×13'], 12)
arrow([(bx, 812), (bx, 790)])
d0, d1 = 596, 790
S.append(f'<rect x="{bx - 150}" y="{d0}" width="300" height="{d1 - d0}" rx="12" fill="{C["box"]}" stroke="#888" stroke-width="1.2"/>')
text(bx - 190, (d0 + d1) / 2, ['Decoder'], 13)
dec = [('ConvT k4 s2  128 → 64', '→ 16×26'), ('ConvT k4 s2  64 → 32', '→ 32×52'), ('ConvT k4 s2  32 → 1', '→ 64×104')]
y = d1 - 10
for i, (a, b) in enumerate(dec):
    y -= 62 if i < 2 else 44
    if i < 2: rect(bx - 120, y + 34, 240, 20, C['norm'], 'BatchNorm · ReLU', 11)
    rect(bx - 120, y, 240, 34, C['conv'], [a, b], 12, True)
    if i < 2: arrow([(bx, y), (bx, y - (8 if i == 0 else 10))])
line([(bx, y), (bx, d0)])
arrow([(bx, d0), (bx, 574)])
text(bx, 556, 'Reconstruction x_rec', 13, True)
arrow([(bx, 540), (bx, 520)])
rect(bx - 150, 462, 300, 58, C['emb'], ['Anomaly score per ROI', 'masked MSE(x, x_rec) over valid frames',
     'REAR: duration-weighted mean of its 2 views'], 12, True)
arrow([(bx, 462), (bx, 440)])
rect(bx - 150, 396, 300, 44, C['attn'], ['Threshold from validation', 'maximize balanced accuracy'], 12, True)
arrow([(bx, 396), (bx, 372)])
text(bx, 350, ['Healthy / Fault per ROI', 'AUC · accuracy · recall · per position / type'], 13, True)

rect(bx - 190, 210, 380, 92, C['loss'], ['Training loss (Healthy views only)', 'L = masked MSE(x, x_rec)',
     'early stop: validation Healthy reconstruction (patience 15)', 'AdamW 1e-3 · batch 64 · ≤ 100 ep · GPU ~50%'], 12, True, dash=True)
arrow([(bx + 190, 556), (bx + 215, 556), (bx + 215, 256), (bx + 190, 256)], dash=True)
S.append(f'<line x1="{bx + 120}" y1="556" x2="{bx + 190}" y2="556" stroke="#222" stroke-width="1.4" stroke-dasharray="4 3"/>')

S.append(f'<line x1="{W / 2 + 15}" y1="80" x2="{W / 2 + 15}" y2="1200" stroke="#ccc" stroke-dasharray="5 5"/>')
legend = [(C['emb'], 'input / score'), (C['conv'], 'convolution'), (C['norm'], 'normalization'),
          (C['attn'], 'bottleneck / decision'), (C['lin'], 'linear'), (C['box'], 'split / container')]
lx = 120
for col, lab in legend:
    S.append(f'<rect x="{lx}" y="1272" width="14" height="14" rx="3" fill="{col}" stroke="#555"/>')
    S.append(f'<text x="{lx + 20}" y="1283" font-size="11.5" fill="#333">{lab}</text>')
    lx += 20 + len(lab) * 6.0 + 22
text(W / 2, 1306, 'Figure: EV per-ROI fault detection – normal-only autoencoder (2026-11-19).', 14)
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Helvetica, Arial, \'DejaVu Sans\', sans-serif">'
       f'<rect width="{W}" height="{H}" fill="white"/>' + ''.join(S) + '</svg>')
OUT.with_suffix('.svg').write_text(svg, encoding='utf-8')
print('ok', OUT)
