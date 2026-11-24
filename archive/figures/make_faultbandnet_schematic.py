"""Simple schematic of FaultBandNet (experiments/fault_net_20261124/train_fault_net.py). -> SVG.
Usage: python make_faultbandnet_schematic.py figures/faultbandnet_schematic
"""
import sys, math
from pathlib import Path
OUT = Path(sys.argv[1]); S = []
W, H = 1640, 800


def text(x, y, s, fs=17, bold=False, color='#222', anchor='middle'):
    S.append(f'<text x="{x}" y="{y}" font-size="{fs}" font-weight="{"700" if bold else "400"}" text-anchor="{anchor}" fill="{color}">{s}</text>')


def box(x, y, w, h, fill, title=None, sub=None, fs=17, rx=10):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="#555" stroke-width="1.4"/>')
    if title: text(x + w / 2, y + h / 2 + (-3 if sub else 6), title, fs, True)
    if sub: text(x + w / 2, y + h / 2 + 18, sub, 13, color='#555')


def arrow(x1, y1, x2, y2, lab=None):
    L = math.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / L, (y2 - y1) / L; bx, by = x2 - ux * 11, y2 - uy * 11
    S.append(f'<line x1="{x1}" y1="{y1}" x2="{bx}" y2="{by}" stroke="#333" stroke-width="2.3"/>')
    S.append(f'<polygon points="{x2},{y2} {bx - uy * 6},{by + ux * 6} {bx + uy * 6},{by - ux * 6}" fill="#333"/>')
    if lab: text((x1 + x2) / 2, (y1 + y2) / 2 - 10, lab, 12, color='#555')


def fmap(x, y, w, h, col, kind):
    """small picture of an angle x frequency map"""
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#2b1a3a" stroke="#555"/>')
    if kind == 'wind': S.append(f'<rect x="{x + w * .35}" y="{y + h * .55}" width="{w * .3}" height="{h * .4}" fill="{col}" opacity=".75"/>')
    if kind == 'pwm': S.append(f'<line x1="{x + w / 2}" y1="{y + h * .15}" x2="{x + w / 2}" y2="{y + h * .45}" stroke="{col}" stroke-width="3"/>')
    if kind == 'whine': S.append(f'<line x1="{x + w / 2}" y1="{y + 4}" x2="{x + w / 2}" y2="{y + h * .5}" stroke="{col}" stroke-width="3"/>')


def block3d(x, cy, h, w, fill):
    d = 8
    S.append(f'<polygon points="{x},{cy - h / 2} {x + d},{cy - h / 2 - d} {x + w + d},{cy - h / 2 - d} {x + w},{cy - h / 2}" fill="{fill}" stroke="#555" opacity=".85"/>')
    S.append(f'<polygon points="{x + w},{cy - h / 2} {x + w + d},{cy - h / 2 - d} {x + w + d},{cy + h / 2 - d} {x + w},{cy + h / 2}" fill="{fill}" stroke="#555" opacity=".65"/>')
    S.append(f'<rect x="{x}" y="{cy - h / 2}" width="{w}" height="{h}" fill="{fill}" stroke="#555" stroke-width="1.3"/>')


text(W / 2, 42, 'FaultBandNet: fault bands × microphone angle → fault type and position', 25, True)
box(30, 345, 170, 100, '#F8D3D7', 'Recording', '3DAR, one rotation')
rows = [(140, 'Wind map', '3.3–6.2 kHz', '36 × 58', 'wind', '#9ad9c5'), (340, 'PWM map', 'around 5 kHz', '36 × 68', 'pwm', '#f4a259'), (540, 'Whine map', 'around order 16', '36 × 51', 'whine', '#8ecae6')]
for y, name, band, size, kind, col in rows:
    arrow(200, 395, 262, y + 55)
    fmap(265, y, 110, 110, col, kind)
    text(320, y - 10, name, 16, True); text(320, y + 128, band, 13, color='#555'); text(320, y + 144, size, 13, color='#555')
    x = 420; arrow(380, y + 55, x - 4, y + 55, 'conv')
    for i, (h, w) in enumerate([(100, 18), (80, 26), (60, 34)]):
        block3d(x, y + 55, h, w, '#C9E6F7')
        if i < 2: arrow(x + w + 12, y + 55, x + w + 50, y + 55, 'conv')
        x += w + 54
    arrow(x - 40, y + 55, 742, 395)
text(320, 742, 'each map: angle (36) × frequency,', 13, color='#555'); text(320, 760, 'relative to neighbours in the recording', 13, color='#555')
text(590, 742, '2-D CNN branch per map', 15, True); text(590, 760, 'pooling along frequency only', 13, color='#555')
# per-angle features strip
S.append(f'<rect x="745" y="285" width="46" height="220" fill="#DCDDF3" stroke="#555" stroke-width="1.3"/>')
for i in range(1, 12): S.append(f'<line x1="745" y1="{285 + i * 220 / 12:.1f}" x2="791" y2="{285 + i * 220 / 12:.1f}" stroke="#888" stroke-width=".7"/>')
text(768, 270, 'features', 15, True); text(768, 525, 'per angle', 13, color='#555'); text(768, 541, '192 × 36', 13, color='#555')
arrow(795, 395, 850, 395)
box(850, 345, 170, 100, '#C9E6F7', '1-D CNN', 'over angle (circular)')
arrow(1020, 395, 1070, 395)
box(1070, 345, 120, 100, '#FCE1C6', 'FC', '→ 9 outputs')
arrow(1190, 395, 1240, 395)
# output grid
gx, gy, cw, ch = 1300, 305, 95, 56
for j, c in enumerate(['Whine', 'PWM', 'Wind']): text(gx + cw * j + cw / 2, gy - 12, c, 15, True)
for i, p in enumerate(['Front', 'Rear', 'Right']):
    text(gx - 10, gy + ch * i + ch / 2 + 5, p, 15, True, anchor='end')
    for j in range(3):
        on = (i, j) == (1, 1)
        S.append(f'<rect x="{gx + cw * j}" y="{gy + ch * i}" width="{cw}" height="{ch}" fill="{"#F8C4C4" if on else "#FFFFFF"}" stroke="#555"/>')
        text(gx + cw * j + cw / 2, gy + ch * i + ch / 2 + 5, '0.93' if on else ['0.02', '0.05', '0.01', '0.04', '', '0.03', '0.02', '0.06', '0.08'][i * 3 + j], 14, on, '#222' if on else '#888')
text(gx + cw * 1.5, gy + ch * 3 + 30, 'Output: probability per position × fault', 15, True)
text(gx + cw * 1.5, gy + ch * 3 + 50, 'example: PWM at the rear speaker', 13, color='#555')
text(W / 2, 96, 'Input uses only the bands where faults appear (from the original KU Leuven files) and keeps the 3DAR angle axis', 15, color='#555')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, DejaVu Sans">'
       f'<rect width="{W}" height="{H}" fill="white"/>' + ''.join(S) + '</svg>')
OUT.with_suffix('.svg').write_text(svg, encoding='utf-8'); print('ok')
