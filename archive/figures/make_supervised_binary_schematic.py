"""Schematic of the supervised Healthy/Fault model (senior-memo structure) as run in
experiments/strict_split_20261123/run_binary_strict.py (= sup_binary_20261119 code, strict split). -> SVG.
Usage: python make_supervised_binary_schematic.py figures/supervised_binary_schematic
"""
import sys, math
from pathlib import Path

OUT = Path(sys.argv[1]); S = []
C = dict(inp='#F8D3D7', conv='#C9E6F7', norm='#F3F5C4', lin='#DCDDF3', out='#CFEBD3', z='#FCE1C6', box='#F3F3F3')


def esc(s): return s.replace('&', '&amp;').replace('<', '&lt;')


def text(x, y, lines, fs=12, bold=False, anchor='middle', color='#222', italic=False):
    if isinstance(lines, str): lines = [lines]
    lh = fs * 1.25; y0 = y - (len(lines) - 1) * lh / 2 + fs * .36
    for i, l in enumerate(lines):
        S.append(f'<text x="{x:.1f}" y="{y0 + i * lh:.1f}" font-size="{fs if i == 0 else fs - 1.5}" '
                 f'font-weight="{"600" if bold and i == 0 else "400"}" text-anchor="{anchor}" fill="{color}"'
                 f'{" font-style=\"italic\"" if italic else ""}>{esc(l)}</text>')


def rect(x, y, w, h, fill, lines, fs=12, bold=True, dash=False):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="#555" stroke-width="1.1"'
             f'{" stroke-dasharray=\"5 4\"" if dash else ""}/>')
    text(x + w / 2, y + h / 2, lines, fs, bold)


def arrow(x1, y1, x2, y2, dash=False):
    L = math.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / L, (y2 - y1) / L; bx, by = x2 - ux * 9, y2 - uy * 9
    S.append(f'<line x1="{x1}" y1="{y1}" x2="{bx}" y2="{by}" stroke="#222" stroke-width="1.5"{" stroke-dasharray=\"4 3\"" if dash else ""}/>')
    S.append(f'<polygon points="{x2},{y2} {bx - uy * 4.5:.1f},{by + ux * 4.5:.1f} {bx + uy * 4.5:.1f},{by - ux * 4.5:.1f}" fill="#222"/>')


def tensor(x, cy, h, w, fill, label, shape):
    d = 8
    S.append(f'<polygon points="{x},{cy - h / 2} {x + d},{cy - h / 2 - d} {x + w + d},{cy - h / 2 - d} {x + w},{cy - h / 2}" fill="{fill}" stroke="#555" opacity=".8"/>')
    S.append(f'<polygon points="{x + w},{cy - h / 2} {x + w + d},{cy - h / 2 - d} {x + w + d},{cy + h / 2 - d} {x + w},{cy + h / 2}" fill="{fill}" stroke="#555" opacity=".6"/>')
    S.append(f'<rect x="{x}" y="{cy - h / 2}" width="{w}" height="{h}" fill="{fill}" stroke="#555"/>')
    text(x + w / 2, cy + h / 2 + 18, label, 12, True); text(x + w / 2, cy + h / 2 + 34, shape, 11, color='#444')


W, H = 1500, 760
text(W / 2, 30, 'Supervised Healthy / Fault model (senior-memo structure) — one angular ROI at a time', 17, True)
text(W / 2, 52, 'experiments/strict_split_20261123/run_binary_strict.py (same code as sup_binary_20261119) · strict source-disjoint split', 11.5, italic=True, color='#666')

# --- row 1: input pipeline ---
y1 = 120
rect(30, y1 - 32, 210, 64, C['inp'], ['Input recording', '1 rotation ≈ 4 s · 48 kHz', '3 speakers + angle CSV'], 12)
arrow(240, y1, 280, y1)
rect(280, y1 - 32, 250, 64, C['inp'], ['Cut 3 angular ROIs (≈1 s)', 'FRONT 135–225° · RIGHT 225–315°', 'REAR 315–360° | 0–45° (2 views)'], 12)
arrow(530, y1, 570, y1)
rect(570, y1 - 32, 250, 64, C['inp'], ['Log-Mel per view', '64 mel · 0–12 kHz · hop 10 ms', '≤ 100 frames (+ mask)'], 12)
arrow(820, y1, 860, y1)
rect(860, y1 - 32, 250, 64, C['norm'], ['Normalize', 'per-mel z-score', '(mean/std from train ROIs)'], 12)
arrow(1110, y1, 1150, y1)
rect(1150, y1 - 32, 320, 64, C['box'], ['Each ROI judged separately', 'no information from the other ROIs', 'label = class of THAT speaker (Healthy / Fault)'], 12)

# --- row 2: network ---
cy = 400
x = 40
tensor(x, cy, 190, 14, C['inp'], 'input view', '1 × 64 × 100'); x0 = x + 30
blocks = [('Conv 3×3 1→32', 32, 150, 24, '32 × 32 × 50'), ('Conv 3×3 32→64', 64, 110, 34, '64 × 16 × 25'),
          ('Conv 3×3 64→128', 128, 75, 46, '128 × 8 × 12')]
x = 150
for i, (op, ch, h, w, shape) in enumerate(blocks):
    arrow(x0, cy, x - 6, cy)
    text((x0 + x) / 2, cy - 40, op.split(' ')[0] + ' ' + op.split(' ')[1], 10)
    text((x0 + x) / 2, cy - 27, op.split(' ', 2)[2], 10)
    text((x0 + x) / 2, cy - 14, 'GN·GELU·Pool2', 10)
    tensor(x, cy, h, w, C['conv'], f'E{i + 1}', shape); x0 = x + w + 14; x += w + 120
S.insert(0, f'<rect x="135" y="{cy - 140}" width="{x - 175}" height="300" rx="12" fill="{C["box"]}" stroke="#888" stroke-dasharray="6 4"/>')
text(135 + (x - 175) / 2, cy - 122, 'CNN encoder (shared for all positions and views)', 12, True)
arrow(x0, cy, x - 40, cy)
rect(x - 40, cy - 40, 150, 80, C['lin'], ['Masked Avg Pool', '→ vector 128', 'REAR: 2 views', 'duration-weighted'], 11.5)
x += 110
arrow(x, cy, x + 40, cy); x += 40
rect(x, cy - 40, 170, 80, C['conv'], ['FCN', 'Linear 128→128', 'GELU · Dropout 0.2', 'Linear 128→64 · GELU'], 11.5)
x += 170
arrow(x, cy, x + 40, cy); x += 40
rect(x, cy - 28, 90, 56, C['z'], ['latent', '64'], 12)
x += 90
arrow(x, cy, x + 40, cy); x += 40
rect(x, cy - 40, 150, 80, C['lin'], ['Decoder', 'Linear 64 → 2', '(classifier,', 'not reconstruction)'], 11.5)
x += 150
arrow(x, cy, x + 30, cy); x += 30
rect(x, cy - 40, 130, 80, C['out'], ['Softmax', 'P(Fault) ≥ 0.5', '→ Fault'], 12)

# --- row 3: training ---
rect(40, 600, 700, 120, '#FFFFFF', ['Training (supervised: sees BOTH Healthy and Fault labels)',
     'loss = cross-entropy (2 classes) per ROI · AdamW lr 1e-4, wd 1e-5',
     'batch 16 recordings (= 48 ROIs) · ≤ 50 epochs · early stop patience 10',
     'checkpoint = best validation macro-F1 · GPU duty ~50%'], 12, dash=True)
rect(780, 600, 680, 120, '#FFFFFF', ['Data (strict split, seed 42 / 43 / 44)',
     'source files assigned to train / val / test; recordings mixing splits dropped',
     'train 713 / 658 / 651 rec (×3 ROIs) · test 88 / 81 / 81 rec · source overlap 0',
     'test AUC 0.949 (hard healthy only 0.829) · accuracy 0.875'], 12, dash=True)

legend = [(C['inp'], 'input'), (C['norm'], 'normalization'), (C['conv'], 'conv / MLP'), (C['lin'], 'pooling / linear'),
          (C['z'], 'latent'), (C['out'], 'output')]
lx = 40
for col, lab in legend:
    S.append(f'<rect x="{lx}" y="{H - 24}" width="14" height="14" rx="3" fill="{col}" stroke="#555"/>')
    S.append(f'<text x="{lx + 20}" y="{H - 13}" font-size="11.5" fill="#333">{lab}</text>'); lx += 20 + len(lab) * 6.3 + 24
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Helvetica, Arial, \'DejaVu Sans\', sans-serif"><rect width="{W}" height="{H}" fill="white"/>' + ''.join(S) + '</svg>')
OUT.with_suffix('.svg').write_text(svg, encoding='utf-8'); print('ok')
