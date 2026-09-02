"""
fig_gr_SiNaK_reversal.py
g(r) parciales Si-Na y Si-K — comparacion AM / IEX2 / REIEX2
Una figura por x_parent = 6, 9, 12  →  fig_gr_SiNaK_iex2_x{x}.pdf/.png

  Si-Na  As-Melted  — linea solida roja
  Si-K   As-Melted  — linea solida negra
  Si-Na  REIEX2     — dots rojos (sin linea)
  Si-K   REIEX2     — dots negros (sin linea)
  Si-Na  IEX2       — ausente (Na=0 en IEX2 — todos Na→K)
  Si-K   IEX2       — linea solida verde

Datos: GR_CN_RINGS/{proto}_x{x}/gr/gr-p/gr-{sp1}_{sp2}.dat
       Columnas: r[Å]  g(r)  CN(r)
       Un directorio por composicion; las 3 replicas ya promediadas en el TRJ.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

HERE = Path(__file__).parent
GR   = HERE / "GR_CN_RINGS"

# ── Paleta ───────────────────────────────────────────────────────────────────
C_RED   = '#c0392b'   # Si-Na (AM + REIEX1)
C_BLACK = '#1a1a1a'   # Si-K AM
C_GREEN = '#148f62'   # Si-K IEX1 — CVD ΔE=7.1 (deutan); distinguible por linestyle

# ── Lector g(r) ──────────────────────────────────────────────────────────────

def load_gr(proto, x, sp1, sp2):
    """Devuelve (r, gr) o (None, None) si el archivo no existe."""
    for s1, s2 in [(sp1, sp2), (sp2, sp1)]:
        p = GR / f"{proto}_x{x}" / "gr" / "gr-p" / f"gr-{s1}_{s2}.dat"
        if p.exists():
            try:
                d = np.loadtxt(p, comments='#')
                if d.ndim == 2 and d.shape[1] >= 2:
                    return d[:, 0], d[:, 1]
            except Exception:
                pass
    return None, None

# ── Figura ───────────────────────────────────────────────────────────────────

R_MAX   = 8.0    # Å — rango a mostrar
STRIDE  = 3      # downsampling para los dots (evitar exceso de puntos)

# ── Helper de figura ──────────────────────────────────────────────────────────

def make_fig(series, x, label_x, out_stem):
    """
    series: lista de dicts con claves proto, sp1, sp2, color, lw, ls, marker, ms, alpha, label
    label_x: string para la anotacion de composicion
    out_stem: nombre base del archivo de salida (sin extension)
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')

    handles, labels = [], []

    for s in series:
        r, gr = load_gr(s['proto'], x, s['sp1'], s['sp2'])
        if r is None:
            continue
        mask = r <= R_MAX
        rv, gv = r[mask], gr[mask]
        if s.get('dots', False):
            rv, gv = rv[::STRIDE], gv[::STRIDE]
        h, = ax.plot(rv, gv,
                     color=s['color'], lw=s.get('lw', 1.8), ls=s.get('ls', '-'),
                     marker=s.get('marker', 'none'), ms=s.get('ms', 0),
                     markeredgewidth=0, alpha=s.get('alpha', 1.0),
                     zorder=s.get('zorder', 3))
        handles.append(h); labels.append(s['label'])

    ax.axhline(1.0, color='#c3c2b7', lw=0.7, ls='--', zorder=0)

    ax.text(0.97, 0.96, label_x, transform=ax.transAxes,
            ha='right', va='top', fontsize=9.5, color='#52514e')

    leg = ax.legend(handles, labels, fontsize=8.5, frameon=True,
                    loc='upper right', bbox_to_anchor=(0.98, 0.88),
                    framealpha=0.92, edgecolor='#e1e0d9')
    leg.get_frame().set_linewidth(0.6)

    ax.set_xlabel('r  (Å)', fontsize=11, color='#0b0b0b')
    ax.set_ylabel('g(r)', fontsize=11, color='#0b0b0b')
    ax.set_xlim(1.5, R_MAX)
    ax.set_ylim(bottom=0)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(1.0))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.25))

    ax.tick_params(axis='both', labelsize=9.5, color='#c3c2b7', labelcolor='#0b0b0b')
    ax.tick_params(which='minor', length=3, color='#c3c2b7')

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color('#c3c2b7')
    ax.spines['bottom'].set_color('#c3c2b7')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    ax.grid(axis='y', color='#e1e0d9', lw=0.5, zorder=0)

    fig.tight_layout(pad=1.2)
    base = Path(__file__).with_name(out_stem)
    fig.savefig(base.with_suffix('.pdf'), dpi=300, bbox_inches='tight', facecolor='#fcfcfb')
    fig.savefig(base.with_suffix('.png'), dpi=150, bbox_inches='tight', facecolor='#fcfcfb')
    print(f"Saved: {out_stem}.pdf / .png")
    plt.close(fig)


# ── IEX2 / REIEX2 ────────────────────────────────────────────────────────────
# IEX2: todos Na→K, so Si-Na IEX2 es ausente (Na=0). Solo Si-K IEX2 (verde).
# REIEX2: reversion selectiva → vuelve a AM_x{parent}; ambas Si-Na y Si-K presentes.

X_LIST_IEX2 = [1, 3, 6, 9, 12]   # x_parent

IEX2_SERIES = [
    dict(proto='AM',     sp1='Si', sp2='Na', color=C_RED,   lw=1.8, ls='-',
         marker='none', ms=0, alpha=1.0, zorder=3, label='Si–Na  As-melted'),
    dict(proto='AM',     sp1='Si', sp2='K',  color=C_BLACK, lw=1.8, ls='-',
         marker='none', ms=0, alpha=1.0, zorder=3, label='Si–K  As-melted'),
    dict(proto='REIEX2', sp1='Si', sp2='Na', color=C_RED,   lw=0,   ls='-',
         marker='o',    ms=3.5, alpha=0.75, zorder=2, dots=True, label='Si–Na  Reverse IEX2'),
    dict(proto='REIEX2', sp1='Si', sp2='K',  color=C_BLACK, lw=0,   ls='-',
         marker='o',    ms=3.5, alpha=0.75, zorder=2, dots=True, label='Si–K  Reverse IEX2'),
    dict(proto='IEX2',   sp1='Si', sp2='K',  color=C_GREEN, lw=1.8, ls='-',
         marker='none', ms=0, alpha=1.0, zorder=3, label='Si–K  IEX2'),
]

for x in X_LIST_IEX2:
    make_fig(IEX2_SERIES, x,
             label_x=f'xₐ = {x} mol% K₂O',
             out_stem=f'fig_gr_SiNaK_iex2_x{x}')
