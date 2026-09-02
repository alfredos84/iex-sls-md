"""
fig_gr_cn_AM.py
Figuras g(r) y CN(r) para As-Melted a 300 K, todos los pares.
Estilo: fig_Ea_vs_x.py — línea+marcador en un solo plot, grid solo eje Y,
        paneles cuadrados lado a lado, sin tick_params top/right.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE    = Path(__file__).parent
OUT_DIR = HERE.parent / "fig_gr_cn_AM"
OUT_DIR.mkdir(exist_ok=True)

X_LIST = [0, 1, 3, 6, 9, 12, 15]

COLORS  = {0: '#1a3a5c', 1: '#1f6fa0', 3: '#2ca0c4',
           6: '#4cbf8f', 9: '#a8d05a', 12: '#e08c2a', 15: '#c0392b'}
MARKERS = {0: 'o', 1: 's', 3: '^', 6: 'D', 9: 'v', 12: 'P', 15: '*'}

PAIRS = [
    ('Si', 'Si'), ('Si', 'O'),  ('Si', 'Ca'), ('Si', 'Na'), ('Si', 'K'),
    ('O',  'O'),  ('O',  'Ca'), ('O',  'Na'), ('O',  'K'),
    ('Ca', 'Ca'), ('Ca', 'Na'), ('Ca', 'K'),
    ('Na', 'Na'), ('Na', 'K'),
    ('K',  'K'),
]


def load_gr(x, sp1, sp2):
    base = HERE / f"AM_x{x}" / "gr"
    for s1, s2 in [(sp1, sp2), (sp2, sp1)]:
        p = base / "gr-p" / f"gr-{s1}_{s2}.dat"
        if p.exists():
            try:
                d = np.loadtxt(p, comments='#')
                if d.ndim == 2 and d.shape[1] >= 2:
                    return d[:, 0], d[:, 1]
            except Exception:
                pass
    return None, None


def load_cn(x, sp1, sp2):
    base = HERE / f"AM_x{x}" / "gr" / "cn"
    for s1, s2 in [(sp1, sp2), (sp2, sp1)]:
        p = base / f"cn-{s1}_{s2}.dat"
        if p.exists():
            try:
                d = np.loadtxt(p, comments='#')
                if d.ndim == 2 and d.shape[1] >= 2:
                    return d[:, 0], d[:, 1]
            except Exception:
                pass
    return None, None


def fig_pair(sp1, sp2):
    # figsize calibrado para paneles ~cuadrados (4.1 × 4.1 in cada uno)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.12, wspace=0.35)

    any_data = False
    handles, labels = [], []

    for x in X_LIST:
        rv, gv = load_gr(x, sp1, sp2)
        if rv is None:
            continue
        rc, cv = load_cn(x, sp1, sp2)

        c = COLORS[x]
        m = MARKERS[x]
        lbl = f'$x={x}$'

        h, = ax1.plot(rv, gv, color=c, lw=1.5, label=lbl, zorder=3)
        if rc is not None:
            ax2.plot(rc, cv, color=c, lw=1.5, zorder=3)

        handles.append(h)
        labels.append(lbl)
        any_data = True

    for ax, title in [(ax1, f'$g(r)$  —  {sp1}–{sp2}'),
                      (ax2, f'$CN(r)$  —  {sp1}–{sp2}')]:
        ax.set_xlabel(r'$r$ (Å)', fontsize=11)
        ax.set_xlim(0, 8.0)
        ax.tick_params(labelsize=9.5)
        ax.grid(axis='y', lw=0.35, color='#dddddd', zorder=0)
        ax.set_title(title, fontsize=11)

    ax1.set_ylabel(r'$g(r)$', fontsize=11)
    ax2.set_ylabel(r'$CN(r)$', fontsize=11)

    if any_data and handles:
        ax1.legend(handles, labels, fontsize=8.5, frameon=True,
                   framealpha=0.9, edgecolor='#cccccc')

    stem = f"fig_gr_cn_AM_{sp1}_{sp2}"
    fig.savefig(OUT_DIR / f"{stem}.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=150, bbox_inches='tight')
    print(f"  {stem}.pdf")
    plt.close(fig)


if __name__ == '__main__':
    print("Generando figuras AM 300K...")
    for sp1, sp2 in PAIRS:
        fig_pair(sp1, sp2)
    print("Listo.")
