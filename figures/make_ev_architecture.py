"""Transformer-figure style schematic of the two Penguin baseline models -> SVG + PNG."""
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

W, H = 1120, 1260


# ================= (a) Localization: LocalizationCNN =================
ax = 300
text(ax, 34, '(a) Fault-source localization (SSL)', 17, True)
text(ax, 56, 'LocalizationCNN · ev3dar/models.py', 11, color='#666', italic=True)

text(ax, 1222, ['Inputs', 'WAV (1 rotation ≈ 4 s, 48 kHz) + encoder angle CSV'], 13, True)
arrow([(ax, 1196), (ax, 1158)])
rect(ax - 120, 1110, 240, 48, C['emb'], ['Log-Mel Spectrogram', '64 mel · 0–12 kHz · n_fft 2048 · hop 480'], 13, True)
arrow([(ax, 1110), (ax, 1078)])
rect(ax - 120, 1030, 240, 48, C['emb'], ['Time → Angle (encoder)', 'center + 1 s context → 128 × 360°'], 13, True)
S.append(f'<circle cx="{ax + 168}" cy="1054" r="17" fill="white" stroke="#222" stroke-width="1.2"/>')
S.append(f'<polygon points="{ax + 168},1037 {ax + 160},1032 {ax + 160},1042" fill="#222"/>')
text(ax + 168, 1092, ['measured', 'angle'], 11)
arrow([(ax, 1030), (ax, 990)])

cy0, cy1 = 690, 990
S.append(f'<rect x="{ax - 150}" y="{cy0}" width="300" height="{cy1 - cy0}" rx="12" fill="{C["box"]}" stroke="#888" stroke-width="1.2"/>')
text(ax - 192, (cy0 + cy1) / 2, ['Circular', '1D CNN', 'Encoder'], 13)
blocks = [('Circular Conv1d  k7', '128 → 128'), ('Circular Conv1d  k5', '128 → 256'),
          ('Dilated Circ. Conv1d  k5, d2', '256 → 256'), ('Circular Conv1d  k3', '256 → 128')]
y = cy1 - 16
for i, (a, b) in enumerate(blocks):
    y -= 66
    rect(ax - 115, y + 30, 230, 22, C['norm'], 'InstanceNorm · ReLU', 12)
    rect(ax - 115, y - 6, 230, 34, C['conv'], [a, b], 12, True)
line([(ax, y - 6), (ax, cy0)])
text(ax + 200, (cy0 + cy1) / 2, ['features', '128 × 360'], 11, color='#666', italic=True)

fy = 660; bl, br = ax - 105, ax + 105
line([(ax, cy0), (ax, fy)]); line([(bl, fy), (br, fy)])
arrow([(bl, fy), (bl, 636)]); arrow([(br, fy), (br, 636)])
rect(bl - 90, 590, 180, 46, C['lin'], ['Global Avg Pool', '128 × 360 → 128'], 12, True)
arrow([(bl, 590), (bl, 566)])
rect(bl - 90, 516, 180, 50, C['lin'], ['Linear 128 → 64', 'ReLU · Dropout 0.3'], 12, True)
arrow([(bl, 516), (bl, 492)])
rect(bl - 90, 456, 180, 36, C['lin'], 'Linear 64 → 3', 12)
arrow([(bl, 456), (bl, 432)])
rect(bl - 90, 398, 180, 34, C['soft'], 'Softmax', 12)
arrow([(bl, 398), (bl, 366)])
text(bl, 346, ['Source Count', '1 / 2 / 3'], 13, True)
rect(br - 90, 586, 180, 50, C['conv'], ['Circular Conv1d k3', '128 → 64 · ReLU'], 12, True)
arrow([(br, 586), (br, 562)])
rect(br - 90, 522, 180, 40, C['lin'], ['Conv1d k1', '64 → 1'], 12, True)
arrow([(br, 522), (br, 498)])
rect(br - 90, 464, 180, 34, C['soft'], 'Sigmoid', 12)
arrow([(br, 464), (br, 440)])
rect(br - 90, 386, 180, 54, C['attn'], ['Circular Peak Picking', '+ Hungarian matching'], 12, True)
arrow([(br, 386), (br, 366)])
text(br, 346, ['DOA Angles', '360° probability map'], 13, True)

rect(ax - 170, 220, 340, 84, C['loss'], ['Training loss', 'L = CE(count) + 5 · Focal(map, target)',
     'target: Gaussian at 180° / 0° / 270°, σ = 5°', 'AdamW 1e-4 · 50 ep · patience 10'], 12, True, dash=True)
arrow([(bl, 322), (bl, 304)], dash=True); arrow([(br, 322), (br, 304)], dash=True)
text(ax, 140, ['3 seeds (42/43/44) · recording split 8:1:1', 'note: every recording = 3 sources, same layout'], 12, color='#444')

# ================= (b) Sound type: SoundTypeCNN =================
bx = 830
text(bx, 34, '(b) Sound-type identification (ID)', 17, True)
text(bx, 56, 'SoundTypeCNN · ev3dar/models.py', 11, color='#666', italic=True)

text(bx, 1222, ['Inputs', 'same WAV + angle CSV'], 13, True)
arrow([(bx, 1196), (bx, 1158)])
rect(bx - 150, 1100, 300, 58, C['emb'], ['Cut 3 angular ROIs', 'FRONT 135–225° · RIGHT 225–315°', 'REAR 315–360° + 0–45° (2 views)'], 12, True)
arrow([(bx, 1100), (bx, 1076)])
rect(bx - 150, 1028, 300, 48, C['emb'], ['Log-Mel per view', '3 pos × 2 views × 64 × 100 (+mask)'], 13, True)
arrow([(bx, 1028), (bx, 990)])

fc0, fc1 = 700, 990
S.append(f'<rect x="{bx - 165}" y="{fc0}" width="330" height="{fc1 - fc0}" rx="12" fill="{C["box"]}" stroke="#888" stroke-width="1.2"/>')
text(bx + 210, (fc0 + fc1) / 2, ['shared', 'for each', 'position'], 13)
y = fc1 - 16
for i, c in enumerate([32, 64, 128]):
    y -= 66
    rect(bx - 130, y + 30, 260, 22, C['norm'], 'GroupNorm · GELU · MaxPool 2', 12)
    rect(bx - 130, y - 6, 260, 34, C['conv'], [f'Conv2d 3×3', f'{[1, 32, 64][i]} → {c}'], 12, True)
y -= 50
rect(bx - 130, y, 260, 40, C['lin'], ['Masked Avg Pool → 128', 'REAR: 2 views duration-weighted'], 12, True)
line([(bx, y), (bx, fc0)])

# concat extras
arrow([(bx, fc0), (bx, 666)])
rect(bx - 130, 626, 260, 40, C['attn'], ['Concat (optional)', '+ PSD 64 · + RPM 2 (estimated)'], 12, True)
arrow([(bx, 626), (bx, 602)])
rect(bx - 130, 532, 260, 70, C['conv'], ['FCN', 'Linear → 128 · GELU · Dropout 0.2', 'Linear 128 → 64 · GELU'], 12, True)
arrow([(bx, 532), (bx, 508)])
text(bx + 175, 520, 'latent 64', 11, color='#666', italic=True)
rect(bx - 130, 470, 260, 38, C['lin'], 'Decoder: Linear 64 → 6', 12)
arrow([(bx, 470), (bx, 446)])
rect(bx - 130, 412, 260, 34, C['soft'], 'Softmax', 12)
arrow([(bx, 412), (bx, 380)])
text(bx, 352, ['Sound type per position (FRONT / REAR / RIGHT)', 'Healthy · Imbalance · Modulation · Whine · Wind · PWM',
               'fault = 1 − P(Healthy) ≥ 0.5'], 13, True)

rect(bx - 175, 220, 350, 84, C['loss'], ['Training loss', 'L = CE(6 classes) per position',
     'best = highest val macro-F1', 'AdamW 1e-4 · 50 ep · patience 10 · batch 16'], 12, True, dash=True)
arrow([(bx, 322), (bx, 304)], dash=True)
text(bx, 140, ['GPU duty limited to ~50% (heat)', 'SSL and ID are trained separately'], 12, color='#444')

S.append(f'<line x1="{W / 2 + 10}" y1="80" x2="{W / 2 + 10}" y2="1200" stroke="#ccc" stroke-dasharray="5 5"/>')
legend = [(C['emb'], 'input / preprocessing'), (C['conv'], 'convolution / MLP'), (C['norm'], 'normalization'),
          (C['attn'], 'matching / concat'), (C['lin'], 'linear / pooling'), (C['soft'], 'output activation')]
lx = 60
for col, lab in legend:
    S.append(f'<rect x="{lx}" y="1252" width="14" height="14" rx="3" fill="{col}" stroke="#555"/>')
    S.append(f'<text x="{lx + 20}" y="1263" font-size="11.5" fill="#333">{lab}</text>')
    lx += 20 + len(lab) * 6.0 + 30
text(W / 2, 1300, 'Figure: 3DAR_EV baseline – model architectures (ev3dar).', 14)

H = 1320
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Helvetica, Arial, \'DejaVu Sans\', sans-serif">'
       '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       '<path d="M0,0 L10,5 L0,10 z" fill="#222"/></marker>'
       '<marker id="ahs" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto">'
       '<path d="M0,0 L10,5 L0,10 z" fill="#222"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="white"/>' + ''.join(S) + '</svg>')
OUT.with_suffix('.svg').write_text(svg, encoding='utf-8')

import sys as _s; _s.exit(0)
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport=dict(width=W, height=H), device_scale_factor=2)
    pg.set_content(f'<html><body style="margin:0">{svg}</body></html>'); pg.screenshot(path=str(OUT.with_suffix('.png')))
    b.close()
print('ok', OUT)
