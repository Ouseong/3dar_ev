"""FaultBandNet model schematic with tensor shapes at every stage (train_fault_net.py). -> SVG.
Usage: python make_faultbandnet_model_io.py figures/faultbandnet_model_io
"""
import sys, math
from pathlib import Path
OUT = Path(sys.argv[1]); S = []
W, H = 1800, 960


def text(x, y, s, fs=16, bold=False, color='#222', anchor='middle'):
    S.append(f'<text x="{x}" y="{y}" font-size="{fs}" font-weight="{"700" if bold else "400"}" text-anchor="{anchor}" fill="{color}">{s}</text>')


def box(x, y, w, h, fill, lines, rx=10, stroke='#555'):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    n = len(lines); y0 = y + h / 2 - (n - 1) * 10 + 5
    for i, l in enumerate(lines):
        if isinstance(l, tuple): s, fs, b, c = l
        else: s, fs, b, c = l, 13, False, '#444'
        text(x + w / 2, y0 + i * 20, s, fs, b, c)


def arrow(x1, y1, x2, y2, lab=None, col='#333'):
    L = math.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / L, (y2 - y1) / L; bx, by = x2 - ux * 10, y2 - uy * 10
    S.append(f'<line x1="{x1}" y1="{y1}" x2="{bx}" y2="{by}" stroke="{col}" stroke-width="2"/>')
    S.append(f'<polygon points="{x2},{y2} {bx - uy * 5},{by + ux * 5} {bx + uy * 5},{by - ux * 5}" fill="{col}"/>')
    if lab: text((x1 + x2) / 2, (y1 + y2) / 2 - 8, lab, 12, color='#555')


def fmap(x, y, w, h, col, kind):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#2b1a3a" stroke="#555"/>')
    if kind == 'wind': S.append(f'<rect x="{x + w * .35}" y="{y + h * .5}" width="{w * .3}" height="{h * .45}" fill="{col}" opacity=".75"/>')
    else: S.append(f'<line x1="{x + w / 2}" y1="{y + 6}" x2="{x + w / 2}" y2="{y + h * .45}" stroke="{col}" stroke-width="3"/>')


def shape(x, y, s): text(x, y, s, 13, True, '#7a2e2e')


text(W / 2, 40, 'FaultBandNet — model (input → output)', 26, True)
text(W / 2, 70, 'shapes are per recording (batch dimension omitted); 145,417 parameters', 15, color='#555')

# ---- input column
box(20, 400, 150, 120, '#F8D3D7', [('Recording', 16, True, '#222'), '3DAR, one rotation', '48 kHz audio', '+ mic angle track'])
box(205, 100, 170, 34, '#EEE', [('Pre-processing (fixed)', 13, True, '#222')], rx=6)
text(290, 155, 'STFT n_fft 8192, hop 480', 12, color='#555'); text(290, 171, '→ average dB per 10° angle bin', 12, color='#555')
text(290, 187, '→ relative inside the recording', 12, color='#555')
rows = [(240, 'Wind map', '3.3–6.2 kHz, 50 Hz bins', 58, 'wind', '#9ad9c5', 29, 14),
        (450, 'PWM map', '4.8–5.2 kHz, 5.9 Hz bins', 68, 'pwm', '#f4a259', 34, 17),
        (660, 'Whine map', 'order 16 ± 150 Hz', 51, 'whine', '#8ecae6', 25, 12)]
for y, name, band, F, kind, col, F2, F3 in rows:
    cy = y + 55
    arrow(170, 460, 222, cy)
    fmap(225, y, 110, 110, col, kind)
    text(280, y - 10, name, 15, True); text(280, y + 128, band, 12, color='#555'); shape(280, y + 146, f'1 × 36 × {F}')
    # branch: conv blocks with shapes
    x = 380
    for i, (c, f) in enumerate([(16, F2), (32, F3), (32, F3)]):
        arrow(x - 40 if i else 340, cy, x, cy)
        hh = 96 - i * 14
        box(x, cy - hh / 2, 128, hh, '#C9E6F7', [('Conv 3×3', 13, True, '#222'), 'GroupNorm, GELU', 'pool freq ÷2' if i < 2 else 'no pool'], rx=6)
        shape(x + 64, cy + hh / 2 + 17, f'{c} × 36 × {f}')
        x += 160
    arrow(x - 40, cy, x, cy)
    box(x, cy - 32, 120, 64, '#E5E1F5', [('mean + max', 13, True, '#222'), 'over frequency'], rx=6)
    shape(x + 60, cy + 50, '64 × 36')
    arrow(x + 120, cy, 990, 505)
text(605, 845, 'one 2-D CNN branch per map (separate weights)', 15, True)
text(605, 865, 'angle axis padded circularly (0° = 360°); pooling only along frequency, so 36 angles are kept', 12, color='#555')

# ---- angle stage
S.append('<rect x="992" y="395" width="44" height="220" fill="#DCDDF3" stroke="#555" stroke-width="1.3"/>')
for i in range(1, 12): S.append(f'<line x1="992" y1="{395 + i * 220 / 12:.1f}" x2="1036" y2="{395 + i * 220 / 12:.1f}" stroke="#888" stroke-width=".7"/>')
text(1014, 380, 'concat', 14, True); shape(1014, 640, '192 × 36')
arrow(1040, 505, 1080, 505)
box(1080, 445, 130, 120, '#C9E6F7', [('1-D Conv ×2', 14, True, '#222'), 'kernel 5 over angle', 'circular, GELU', 'dropout 0.3'])
shape(1145, 585, '64 × 36')
arrow(1210, 505, 1250, 505)
box(1250, 465, 100, 80, '#FCE1C6', [('Flatten', 14, True, '#222'), 'dropout 0.3'])
shape(1300, 565, '2304')
arrow(1350, 505, 1390, 505)
box(1390, 465, 100, 80, '#FCE1C6', [('Linear', 14, True, '#222'), '2304 → 9'])
shape(1440, 565, '9 → 3 × 3')
arrow(1490, 505, 1530, 505); text(1510, 540, 'sigmoid', 12, color='#555')

# ---- output grid
gx, gy, cw, ch = 1590, 420, 66, 50
for j, c in enumerate(['Whine', 'PWM', 'Wind']): text(gx + cw * j + cw / 2, gy - 10, c, 14, True)
vals = ['0.02', '0.05', '0.01', '0.04', '0.93', '0.03', '0.02', '0.06', '0.08']
for i, p in enumerate(['Front', 'Rear', 'Right']):
    text(gx - 8, gy + ch * i + ch / 2 + 5, p, 14, True, anchor='end')
    for j in range(3):
        on = (i, j) == (1, 1)
        S.append(f'<rect x="{gx + cw * j}" y="{gy + ch * i}" width="{cw}" height="{ch}" fill="{"#F8C4C4" if on else "#FFF"}" stroke="#555"/>')
        text(gx + cw * j + cw / 2, gy + ch * i + ch / 2 + 5, vals[i * 3 + j], 13, on, '#222' if on else '#888')
text(gx + cw * 1.5, gy + ch * 3 + 26, 'Output 3 × 3 probabilities', 14, True)
text(gx + cw * 1.5, gy + ch * 3 + 44, '(position × fault type)', 12, color='#555')
text(gx + cw * 1.5, gy + ch * 3 + 62, '≥ 0.5 → fault there', 12, color='#555')
text(gx + cw * 1.5, gy + ch * 3 + 80, 'example: PWM at rear', 12, color='#555')

# ---- legend for decisions
box(1080, 700, 690, 150, '#FAFAFA', [('How the 3 × 3 output is read', 14, True, '#222'),
    'speaker (sector) is faulty  ←  max over the 3 fault types ≥ 0.5',
    'recording has fault X  ←  max over the 3 positions ≥ 0.5',
    'position of fault X  ←  position with the highest probability',
    'Imbalance / Modulation are not outputs (labelled 0 = healthy)'], stroke='#bbb')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, DejaVu Sans">'
       f'<rect width="{W}" height="{H}" fill="white"/>' + ''.join(S) + '</svg>')
OUT.with_suffix('.svg').write_text(svg, encoding='utf-8'); print('ok')
