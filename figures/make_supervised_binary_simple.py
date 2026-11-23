"""Simple (few words) schematic of the supervised Healthy/Fault model, with tensor sizes. -> SVG.
Usage: python make_supervised_binary_simple.py figures/supervised_binary_simple
"""
import sys, math
from pathlib import Path

OUT = Path(sys.argv[1]); S = []
W, H, CY = 1640, 480, 215


def text(x, y, s, fs=20, bold=False, color='#222'):
    S.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{fs}" font-weight="{"700" if bold else "400"}" '
             f'text-anchor="middle" fill="{color}">{s}</text>')


def arrow(x1, x2, y=CY, lab=None):
    S.append(f'<line x1="{x1}" y1="{y}" x2="{x2 - 12}" y2="{y}" stroke="#333" stroke-width="3"/>')
    S.append(f'<polygon points="{x2},{y} {x2 - 14},{y - 8} {x2 - 14},{y + 8}" fill="#333"/>')
    if lab: text((x1 + x2) / 2, y - 14, lab, 13, color='#555')


def box(x, w, h, fill):
    S.append(f'<rect x="{x}" y="{CY - h / 2}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="#555" stroke-width="1.5"/>')


def label(x, title, sub):
    text(x, CY + 125, title, 21, True); text(x, CY + 152, sub, 16, color='#555')


def block3d(x, h, w, fill, size):
    d = 9
    S.append(f'<polygon points="{x},{CY - h / 2} {x + d},{CY - h / 2 - d} {x + w + d},{CY - h / 2 - d} {x + w},{CY - h / 2}" fill="{fill}" stroke="#555" opacity=".85"/>')
    S.append(f'<polygon points="{x + w},{CY - h / 2} {x + w + d},{CY - h / 2 - d} {x + w + d},{CY + h / 2 - d} {x + w},{CY + h / 2}" fill="{fill}" stroke="#555" opacity=".65"/>')
    S.append(f'<rect x="{x}" y="{CY - h / 2}" width="{w}" height="{h}" fill="{fill}" stroke="#555" stroke-width="1.5"/>')
    text(x + w / 2, CY + h / 2 + 24, size, 14, color='#333')


# 1. recording
x = 30; box(x, 150, 110, '#F8D3D7')
pts = ' '.join(f'{x + 15 + i * 2:.0f},{CY + 30 * math.sin(i / 3) * math.exp(-((i - 30) / 25) ** 2):.0f}' for i in range(61))
S.append(f'<polyline points="{pts}" fill="none" stroke="#a33" stroke-width="2"/>')
label(x + 75, 'Recording', 'one mic rotation · 4 s')
arrow(190, 240)

# 2. cut by direction
cx = 310; r = 55
def wedge(a0, a1, col):
    p = lambda a: (cx + r * math.cos(math.radians(a)), CY - r * math.sin(math.radians(a)))
    (x0, y0), (x1, y1) = p(a0), p(a1)
    S.append(f'<path d="M{cx},{CY} L{x0:.1f},{y0:.1f} A{r},{r} 0 0 0 {x1:.1f},{y1:.1f} Z" fill="{col}" stroke="#555"/>')
wedge(135, 225, '#FCE1C6'); wedge(225, 315, '#CFEBD3'); wedge(315, 405, '#DCDDF3'); wedge(45, 135, '#FFFFFF')
for a, t in [(180, 'F'), (270, 'R'), (0, 'B')]:
    text(cx + 32 * math.cos(math.radians(a)), CY - 32 * math.sin(math.radians(a)) + 6, t, 16, True)
label(cx, 'Cut by direction', 'front · right · rear (1 s each)')
arrow(380, 430)

# 3. spectrogram
x = 435; box(x, 150, 110, '#2b1a3a')
for i, y in enumerate([CY + 35, CY + 25, CY + 12, CY - 5]):
    S.append(f'<line x1="{x + 8}" y1="{y}" x2="{x + 142}" y2="{y - 6}" stroke="{["#f9d77e", "#f4a259", "#e76f51", "#b5838d"][i]}" stroke-width="{4 - i * 0.7}"/>')
label(x + 75, 'Spectrogram', 'log-mel')
text(x + 75, CY + 80, '1 × 64 × 100', 14, color='#333')
arrow(595, 655, lab='conv')

# 4. CNN: shrinking feature maps
x = 660; xs = []
for i, (h, w, size) in enumerate([(130, 22, '32 × 32 × 50'), (95, 34, '64 × 16 × 25'), (62, 48, '128 × 8 × 12')]):
    block3d(x, h, w, '#C9E6F7', size); xs.append(x + w / 2)
    if i < 2: arrow(x + w + 14, x + w + 88, lab='conv')
    x += w + 92
label(sum(xs) / 3, 'CNN (3 blocks)', 'smaller map, more channels')
arrow(x - 70, x - 10, lab='pool')

# 5. FCN: vector shrinking 128 -> 64 -> 2
x -= 5
for i, (n, size) in enumerate([(10, '128'), (6, '64'), (2, '2')]):
    h = n * 14
    for k in range(n):
        S.append(f'<rect x="{x}" y="{CY - h / 2 + k * 14}" width="22" height="12" fill="{["#DCDDF3", "#FCE1C6", "#CFEBD3"][i]}" stroke="#555"/>')
    text(x + 11, CY + h / 2 + 24, size, 14, color='#333')
    if i < 2: arrow(x + 30, x + 92, lab='FCN' if i == 0 else 'linear')
    x += 100
label(x - 150, 'Vector → latent → output', '128 → 64 → 2')
arrow(x - 70, x - 15)

# 6. decision
x -= 10
for j, (t, col) in enumerate([('Healthy', '#CFEBD3'), ('Fault', '#F8C4C4')]):
    S.append(f'<rect x="{x}" y="{CY - 58 + j * 64}" width="150" height="52" rx="10" fill="{col}" stroke="#555" stroke-width="1.5"/>')
    text(x + 75, CY - 24 + j * 64, t, 21, True)
label(x + 75, 'Decision', 'per direction')

text(W / 2, 55, 'Supervised Healthy / Fault model (senior-memo structure)', 27, True)
text(W / 2, 438, 'Training: sees both Healthy and Fault labels   |   Test: unseen source sounds (strict split)', 17, color='#555')
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Helvetica, Arial, DejaVu Sans"><rect width="{W}" height="{H}" fill="white"/>' + ''.join(S) + '</svg>')
OUT.with_suffix('.svg').write_text(svg, encoding='utf-8'); print('ok', x + 160)
