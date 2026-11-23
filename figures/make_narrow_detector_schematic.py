"""Simple schematic of the narrow-feature fault detector (experiments/wind_detect_20261124/fault_detector.py). -> SVG.
Usage: python make_narrow_detector_schematic.py figures/narrow_detector_schematic
"""
import sys, math
from pathlib import Path
OUT = Path(sys.argv[1]); S = []
W, H = 1500, 640


def text(x, y, s, fs=18, bold=False, color='#222', anchor='middle'):
    S.append(f'<text x="{x}" y="{y}" font-size="{fs}" font-weight="{"700" if bold else "400"}" text-anchor="{anchor}" fill="{color}">{s}</text>')


def box(x, y, w, h, fill, title, sub=None, fs=19):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#555" stroke-width="1.5"/>')
    text(x + w / 2, y + h / 2 + (-4 if sub else 7), title, fs, True)
    if sub: text(x + w / 2, y + h / 2 + 20, sub, 14, color='#555')


def arrow(x1, y1, x2, y2):
    L = math.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / L, (y2 - y1) / L; bx, by = x2 - ux * 12, y2 - uy * 12
    S.append(f'<line x1="{x1}" y1="{y1}" x2="{bx}" y2="{by}" stroke="#333" stroke-width="2.5"/>')
    S.append(f'<polygon points="{x2},{y2} {bx - uy * 7},{by + ux * 7} {bx + uy * 7},{by - ux * 7}" fill="#333"/>')


text(W / 2, 44, 'Narrow-feature fault detector: measure one known component per fault', 26, True)
box(40, 250, 190, 110, '#F8D3D7', 'Recording', '3DAR, one rotation')
rows = [(120, '#C9E6F7', 'Whine', 'order-16 line vs. neighbours', '~2 kHz, follows RPM'),
        (270, '#FCE1C6', 'PWM', '5 kHz line vs. neighbours', 'fixed frequency'),
        (420, '#D6EFE6', 'Wind', '4.0–5.5 kHz bump shape', 'while mic faces the speaker')]
for y, col, name, what, note in rows:
    arrow(230, 305, 300, y + 50)
    box(300, y, 330, 100, col, f'{name}: {what}', note, 18)
    arrow(630, y + 50, 700, y + 50)
box(700, 120, 190, 100, '#F3F5C4', 'threshold', 'one number')
box(700, 270, 190, 100, '#F3F5C4', 'threshold', 'one number')
box(700, 420, 190, 100, '#F3F5C4', 'small model', 'logistic regression')
for y, name in [(120, 'Whine?'), (270, 'PWM?'), (420, 'Wind?')]:
    arrow(890, y + 50, 960, y + 50); box(960, y + 10, 150, 80, '#FFFFFF', name, 'yes / no')
    arrow(1110, y + 50, 1180, 320)
box(1180, 270, 280, 100, '#CFEBD3', 'Fault in recording?', 'and which type')
box(1180, 430, 280, 80, '#DCDDF3', 'PWM direction', 'sector with the strongest 5 kHz line')
arrow(1110, 330, 1180, 470)
text(W / 2, 580, 'Feature bands come from the ORIGINAL KU Leuven files.   Thresholds / Wind model: fitted on training folds, applied to held-out recordings.', 16, color='#555')
text(W / 2, 606, 'Not covered: Imbalance (lost in low-frequency noise), Modulation (hard even in the originals).', 16, color='#555')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, DejaVu Sans">'
       f'<rect width="{W}" height="{H}" fill="white"/>' + ''.join(S) + '</svg>')
OUT.with_suffix('.svg').write_text(svg, encoding='utf-8'); print('ok')
