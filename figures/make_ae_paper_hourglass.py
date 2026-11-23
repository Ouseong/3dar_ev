"""Hourglass schematic of the paper autoencoder (Kunte et al. Fig. 10, n=6, f=8) on one EV ROI view
(1 x 64 mel x 128 frames). Box height ~ log2(spatial size), width ~ channels. -> SVG.
Usage: python make_ae_paper_hourglass.py figures/ae_paper_hourglass [freq_rows=64 (mel) | 1536 (stft)]
"""
import sys, math
from pathlib import Path

OUT = Path(sys.argv[1])
FR = int(sys.argv[2]) if len(sys.argv) > 2 else 64
STFT = FR != 64
N, F = 6, 8
S = []
COL = dict(inp='#F8D3D7', enc='#C9E6F7', z='#FCE1C6', dec='#D6EFE6', out='#CFEBD3')


def esc(s): return s.replace('&', '&amp;').replace('<', '&lt;')


def text(x, y, s, fs=12, w='400', anchor='middle', color='#222', italic=False):
    S.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{fs}" font-weight="{w}" text-anchor="{anchor}" '
             f'fill="{color}"{" font-style=\"italic\"" if italic else ""}>{esc(s)}</text>')


def arrow(x1, x2, y):
    S.append(f'<line x1="{x1}" y1="{y}" x2="{x2 - 8}" y2="{y}" stroke="#222" stroke-width="1.4"/>')
    S.append(f'<polygon points="{x2},{y} {x2 - 9},{y - 4.5} {x2 - 9},{y + 4.5}" fill="#222"/>')


# tensors: (channels, H, W, colour, label)
T = [(1, FR, 128, 'inp', 'input x')]
for i in range(1, N + 1): T.append((F, FR >> i, 128 >> i, 'z' if i == N else 'enc', 'z' if i == N else f'E{i}'))
for i in range(N - 1, 0, -1): T.append((F, FR >> i, 128 >> i, 'dec', f'D{N - i}'))
T.append((1, FR, 128, 'out', 'x_rec'))

W, H, cy = 1500, 560, 250
gap = 56
def bh(h): return 26 + math.log2(h) * 216 / math.log2(FR)   # 1 -> 26 px, FR -> 242 px
def bw(c): return 16 if c == 1 else 46

xs, x = [], 60
for c, h, w, col, lab in T:
    xs.append(x); x += bw(c) + gap
W = x - gap + 60

ops_enc = 'Conv3×3 · BN · ReLU · MaxPool2'
for k, (c, h, w, col, lab) in enumerate(T):
    x0, ww, hh = xs[k], bw(c), bh(h)
    d = 10 if c > 1 else 4                          # pseudo-3D depth
    S.append(f'<polygon points="{x0},{cy - hh / 2} {x0 + d},{cy - hh / 2 - d} {x0 + ww + d},{cy - hh / 2 - d} {x0 + ww},{cy - hh / 2}" '
             f'fill="{COL[col]}" stroke="#555" stroke-width="1" opacity="0.8"/>')
    S.append(f'<polygon points="{x0 + ww},{cy - hh / 2} {x0 + ww + d},{cy - hh / 2 - d} {x0 + ww + d},{cy + hh / 2 - d} {x0 + ww},{cy + hh / 2}" '
             f'fill="{COL[col]}" stroke="#555" stroke-width="1" opacity="0.6"/>')
    S.append(f'<rect x="{x0}" y="{cy - hh / 2}" width="{ww}" height="{hh}" fill="{COL[col]}" stroke="#555" stroke-width="1.1"/>')
    text(x0 + ww / 2, cy + 142, lab, 12.5, '600')
    text(x0 + ww / 2, cy + 160, f'{c}×{h}×{w}', 11, color='#444')
    if k < len(T) - 1:
        ax0 = x0 + ww + d + 4; ax1 = xs[k + 1] - 4
        arrow(ax0, ax1, cy)

# op labels between tensors (above arrows)
def op_label(k, lines):
    xm = (xs[k] + bw(T[k][0]) + 10 + xs[k + 1]) / 2
    for j, l in enumerate(lines): text(xm, cy - 14 - 13 * (len(lines) - 1 - j), l, 9.5, color='#333')

for k in range(N): op_label(k, ['Conv', 'BN·ReLU', 'Pool↓2'])
for k in range(N, 2 * N - 1): op_label(k, ['ConvT↑2', 'Conv', 'BN·ReLU'])
op_label(2 * N - 1, ['ConvT↑2', 'Conv→1', 'Sigmoid'])

# brackets
def bracket(x0, x1, y, lab, sub):
    S.append(f'<path d="M{x0},{y + 8} L{x0},{y} L{x1},{y} L{x1},{y + 8}" fill="none" stroke="#666" stroke-width="1.2"/>')
    text((x0 + x1) / 2, y - 22, lab, 14, '600'); text((x0 + x1) / 2, y - 6, sub, 11, color='#666', italic=True)

top = cy - bh(FR) / 2 - 30
bracket(xs[0] + 8, xs[N] + bw(F) / 2, top, 'Encoder', f'n = {N} blocks: Conv2d 3×3 ({F} filters) → BatchNorm → ReLU → MaxPool 2×2')
bracket(xs[N] + bw(F) / 2, xs[-1] + 8, top, 'Decoder', f'n−1 = {N - 1} blocks: ConvTranspose2d → Conv2d → BN → ReLU, then output block')
xz = xs[N] + bw(F) / 2
zb = cy + bh(FR >> N) / 2 + 18
text(xz, zb, 'latent', 11, '600', color='#B25E00'); text(xz, zb + 14, f'{F * (FR >> N) * (128 >> N)} values', 10.5, color='#B25E00')

text(W / 2, 30, f'Paper CNN autoencoder (Kunte et al., Fig. 10) on one EV ROI view · n = {N}, f = {F} · 7,753 parameters'
     + (' · linear STFT input' if STFT else ''), 16, '600')
notes = [('Input: linear STFT, n_fft 6144 (7.8 Hz bins), 0–12 kHz = 1536 rows (no mel) × 100 frames (hop 10 ms), clip −120…0 dBFS → [0, 1], pad to 128 frames.'
          if STFT else 'Input: 64 mel × 100 frames, clip −120…0 dBFS → [0, 1], padded to 128 frames (padding masked out of the loss).'),
         'Train on Healthy views only: masked MSE, Adam 1e-4, batch 8, ≤ 100 epochs. Anomaly score = reconstruction error per ROI.',
         'Not in the paper (our choice): ConvTranspose2d kernel 2 stride 2; last Conv2d has 1 output channel. Box height ~ log2(freq rows), not to scale.']
for j, n in enumerate(notes): text(60, H - 92 + 20 * j, n, 12, anchor='start', color='#444')

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Helvetica, Arial, \'DejaVu Sans\', sans-serif"><rect width="{W}" height="{H}" fill="white"/>'
       + ''.join(S) + '</svg>')
OUT.with_suffix('.svg').write_text(svg, encoding='utf-8')
print('ok', OUT, W, H)
