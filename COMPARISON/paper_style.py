"""
paper_style.py
Estilo unificado para las figuras del paper (ver paper_writing/figures_paper/).

- Cada panel ("box") es cuadrado (1:1), mismo tamaño fisico en todas las figuras.
- Area total de la figura escala con la cantidad de paneles: 1 panel = BOX x BOX,
  2 paneles = 2*BOX x BOX (o BOX x 2*BOX), 4 paneles (2x2) = 2*BOX x 2*BOX.
- Fuente: 18 pt en xlabel/ylabel; 16 pt en ticks, leyenda y textos/anotaciones.
- Vocabulario: IEX/IEX1 -> IOX ; As-Melted / No IEX -> AM ; 10CaO -> Ca ; Ca-free igual.
"""

import re
import matplotlib.pyplot as plt

BOX = 5.5          # pulgadas, lado de cada panel cuadrado
FS_LABEL = 18       # xlabel, ylabel
FS_TEXT = 16        # ticks, leyenda, textos/anotaciones, letras de panel


def apply_rcparams():
    plt.rcParams.update({
        "font.family":     "Times New Roman",
        "font.size":       FS_TEXT,
        "axes.linewidth":  0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def relabel(s: str) -> str:
    """Aplica el vocabulario unificado del paper a un string de figura."""
    s = s.replace("No IEX1", "AM").replace("No IEX", "AM")
    s = s.replace("As-Melted", "AM").replace("As-melted", "AM")
    s = s.replace("Ion-Exchanged", "IOX")
    s = s.replace("IEX1", "IOX").replace("IEX", "IOX")
    s = re.sub(r"\b10CaO\b", "Ca", s)
    return s


def make_fig(nrows, ncols, box=BOX):
    """Figura con paneles cuadrados de lado `box`; area total ~ nrows*ncols*box^2."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(box * ncols, box * nrows))
    import numpy as np
    for ax in np.atleast_1d(axes).ravel():
        ax.set_box_aspect(1)
    return fig, axes


def style_axes(ax):
    ax.tick_params(labelsize=FS_TEXT, which="both")
    ax.grid(lw=0.35, color="#dddddd", zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)


def panel_letter(ax, letter, x=0.03, y=0.97):
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top",
            fontsize=FS_TEXT, fontweight="bold", zorder=6)
