"""FaultBandNet training / evaluation procedure (3DAR_EV_FaultBandNet.ipynb: repeated source-disjoint splits). -> SVG.
Usage: python make_faultbandnet_training.py figures/faultbandnet_training
"""
import sys, math
from pathlib import Path
OUT = Path(sys.argv[1]); S = []
W, H = 1700, 870


def text(x, y, s, fs=15, bold=False, color='#222', anchor='middle'):
    S.append(f'<text x="{x}" y="{y}" font-size="{fs}" font-weight="{"700" if bold else "400"}" text-anchor="{anchor}" fill="{color}">{s}</text>')


def box(x, y, w, h, fill, lines, rx=10, stroke='#555', dash=False):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"{" stroke-dasharray=\"6 4\"" if dash else ""}/>')
    n = len(lines); y0 = y + h / 2 - (n - 1) * 10 + 5
    for i, l in enumerate(lines):
        if isinstance(l, tuple): s, fs, b = l
        else: s, fs, b = l, 13, False
        text(x + w / 2, y0 + i * 20, s, fs, b, '#222' if b else '#444')


def arrow(x1, y1, x2, y2, lab=None, col='#333', lx=None):
    L = math.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / L, (y2 - y1) / L; bx, by = x2 - ux * 10, y2 - uy * 10
    S.append(f'<line x1="{x1}" y1="{y1}" x2="{bx}" y2="{by}" stroke="{col}" stroke-width="2"/>')
    S.append(f'<polygon points="{x2},{y2} {bx - uy * 5},{by + ux * 5} {bx + uy * 5},{by - ux * 5}" fill="{col}"/>')
    if lab: text(lx if lx else (x1 + x2) / 2 + 6, (y1 + y2) / 2 - 6, lab, 12, color='#555', anchor='start' if lx is None and x1 == x2 else 'middle')


text(W / 2, 40, 'FaultBandNet — training and evaluation', 26, True)
text(W / 2, 70, 'each tested recording is predicted only by models that never heard any of its 3 source files', 15, color='#555')

# ---- data and labels (left column)
box(30, 110, 300, 110, '#F8D3D7', [('Data: all 1,200 recordings', 15, True), 'Imbalance / Modulation → healthy (out of band)', 'Whine/PWM/Wind count 0/1/2/3: 479/394/244/83', 'input = 3 maps (Wind, PWM, Whine)'])
arrow(180, 220, 180, 260)
box(30, 260, 300, 150, '#FFF4D6', [('Label: 3 × 3 (position × fault)', 15, True), '1 if that speaker plays', 'Whine / PWM / Wind, else 0',
    'e.g. rear = PWM → [0 0 0 / 0 1 0 / 0 0 0]', '3x1 variant: max over types → [0 / 1 / 0]'])
arrow(330, 165, 420, 165)

# ---- split
box(420, 110, 380, 110, '#E5E1F5', [('Source-disjoint split, 30 repeats', 15, True), 'test: ~5.3 % of each fault count, all 3 sources in T', 'train pool: no source file in T', 'same share kept for each fault count (rest unused)'])
fx, fy, fw = 440, 245, 66
for k in range(5):
    S.append(f'<rect x="{fx + k * (fw + 4)}" y="{fy}" width="{fw}" height="34" fill="{"#F8C4C4" if k == 0 else ("#EEEEEE" if k == 4 else "#C9E6F7")}" stroke="#555"/>')
    text(fx + k * (fw + 4) + fw / 2, fy + 22, ['test', 'train', 'train', 'train', 'unused'][k], 13, k == 0)
text(610, fy + 54, 'per repeat: train 329–493 (mean 419), validation ~74, test 63', 12, color='#555')
arrow(560, 300, 560, 340)
box(420, 340, 380, 70, '#E5E1F5', [('train pool → 85 % train / 15 % validation', 14, True), 'per fault count; validation only for early stopping'])

# ---- training loop
arrow(610, 410, 610, 450)
S.append('<rect x="380" y="450" width="830" height="380" rx="16" fill="#F4F9FD" stroke="#7aa6c2" stroke-width="1.6" stroke-dasharray="8 5"/>')
text(395, 475, 'repeated for each of 30 splits', 14, True, '#2f5f80', 'start')
box(410, 495, 230, 90, '#FFF', [('Normalise inputs', 14, True), 'mean / std from', 'training recordings only'])
arrow(640, 540, 690, 540)
box(690, 495, 230, 90, '#C9E6F7', [('Mini-batch (32)', 14, True), '3 maps → FaultBandNet', '→ 3 × 3 logits'])
arrow(920, 540, 970, 540)
box(970, 495, 220, 90, '#FCE1C6', [('Loss: BCE', 14, True), 'positive weight per output', '= #neg / #pos (max 20)'])
arrow(1080, 585, 1080, 625)
box(970, 625, 220, 70, '#FCE1C6', [('AdamW', 14, True), 'lr 1e-3, weight decay 1e-3'])
arrow(970, 660, 920, 660)
box(690, 625, 230, 70, '#DFF3E4', [('end of epoch', 14, True), 'validation AUC', '(3x3: per type, 3x1: per sector)'])
arrow(690, 660, 640, 660)
box(410, 615, 230, 90, '#DFF3E4', [('keep best epoch', 14, True), 'early stop: no gain 15 epochs', 'max 80 epochs'])
S.append('<polyline points="525,615 525,603 805,603 805,595" fill="none" stroke="#333" stroke-width="2"/>'); arrow(805, 600, 805, 586)
text(665, 597, 'next epoch', 12, color='#555')
arrow(525, 705, 525, 745)
box(410, 745, 300, 60, '#F8C4C4', [('predict the test set', 14, True), 'with the best-epoch weights'])
text(1195, 820, 'GPU throttled to ~50 % (Spark heat limit)', 12, color='#555', anchor='end')
arrow(710, 775, 1260, 775)
box(1260, 715, 410, 120, '#FFF4D6', [('Test probabilities', 15, True), 'averaged over the repeats a recording was tested', '1,158 of 1,200 recordings tested', 'test_probabilities_3x3.npy / _3x1.npy'])
arrow(1465, 715, 1465, 665)
box(1260, 415, 410, 250, '#FAFAFA', [('Evaluation (threshold 0.5, fixed)', 15, True),
    'per recording: has fault X? → detection rate,', 'false alarms on healthy 11/09, healthy 11/02,', 'and on other-fault recordings',
    'per sector: is this speaker faulty?', '→ healthy / fault sectors right, by fault count 0–3',
    'position: highest-probability position', 'among recordings with one fault X'], stroke='#bbb')
box(1260, 110, 410, 150, '#FAFAFA', [('Why not k-fold?', 15, True), 'healthy source files are reused up to 12 times,', 'at several positions → no clean k-fold exists', 'earlier 5-fold grouped only the right speaker:', '98 % of test recordings shared a source with training'], stroke='#bbb', dash=True)
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, DejaVu Sans">'
       f'<rect width="{W}" height="{H}" fill="white"/>' + ''.join(S) + '</svg>')
OUT.with_suffix('.svg').write_text(svg, encoding='utf-8'); print('ok')
