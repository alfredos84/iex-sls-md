"""
fig_stress_av_mean_fwhm.py
<P> vs chi promediado sobre las 3 replicas, con el ancho de la distribucion como sombra.

Para cada chi, especie y estado (AM / IEX) y para cada replica (r=1,2,3):
  - media de la presion hidrostatica atomica P_i = -(sxx+syy+szz)/(3*V_i) [GPa]
  - ancho a media altura (FWHM) de la KDE de la distribucion de P_i
Curva  = media de las 3 medias.
Numero junto a cada punto = sigma_<P> = FWHM promediado sobre las 3 replicas (GPa).

Layout 2x2 como fig_stress_mean: (modificadores | red) x (CaO | Ca-free).
Ejes y: modificadores -5..4 GPa ; red con eje partido: O 15..35, Si -130..-100 GPa.
Paneles 2:1 (ancho:alto). Leyenda unica arriba, rotulos (a)-(d).
Atom types: Si=1, O=2, Ca=3, Na=4, K=5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgb
from pathlib import Path
from scipy.stats import gaussian_kde

HERE     = Path(__file__).parent
DUMP_DIR = HERE / "dumps"
VORO_DIR = HERE.parent / "VORONOI"

TYPE_SI, TYPE_O, TYPE_CA, TYPE_NA, TYPE_K = 1, 2, 3, 4, 5
REPLICAS = (1, 2, 3)

XX_CONFIG = {
    20:  (3,  3,  3,  5,  5,  5),
    40:  (6,  6,  6,  10, 10, 10),
    60:  (9,  9,  9,  15, 15, 15),
    80:  (12, 12, 12, 20, 20, 20),
    100: (0,  15, 15, 0,  25, 25),
}
CHI = np.array(sorted(XX_CONFIG), dtype=float)

YLIM_MOD = (-5, 4)
YLIM_O = (15, 35)
YLIM_SI = (-130, -100)


def read_columns(path, ncols):
    data = {}
    in_atoms = False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("ITEM: ATOMS"):
                in_atoms = True
                continue
            if s.startswith("ITEM:"):
                in_atoms = False
                continue
            if in_atoms and s:
                p = s.split()
                data[int(p[0])] = (int(p[1]),) + tuple(float(x) for x in p[2:2 + ncols])
    return data


def pressure_per_type(tag):
    stress = read_columns(DUMP_DIR / f"{tag}_stress.dat", 3)
    vol = read_columns(VORO_DIR / f"{tag}_vol.dat", 1)
    by_type = {t: [] for t in (TYPE_SI, TYPE_O, TYPE_CA, TYPE_NA, TYPE_K)}
    for aid, (t, sxx, syy, szz) in stress.items():
        if aid not in vol or t not in by_type:
            continue
        v = vol[aid][1]
        if v <= 0:
            continue
        by_type[t].append(-(sxx + syy + szz) / (3.0 * v) * 1e-4)   # GPa
    return {t: np.array(vals) for t, vals in by_type.items()}


def fwhm(arr):
    kde = gaussian_kde(arr, bw_method="silverman")
    pad = 0.1 * (arr.max() - arr.min())
    grid = np.linspace(arr.min() - pad, arr.max() + pad, 4000)
    y = kde(grid)
    i = int(np.argmax(y))
    half = y[i] / 2.0
    j = i
    while j > 0 and y[j] >= half:
        j -= 1
    k = i
    while k < len(y) - 1 and y[k] >= half:
        k += 1
    left = grid[j] + (half - y[j]) * (grid[j + 1] - grid[j]) / (y[j + 1] - y[j])
    right = grid[k - 1] + (y[k - 1] - half) * (grid[k] - grid[k - 1]) / (y[k - 1] - y[k])
    return right - left


# stats[sys][state][t] = (mean[chi], fwhm[chi])  (NaN si la especie no existe en ese chi)
stats = {s: {st: {t: (np.full(len(CHI), np.nan), np.full(len(CHI), np.nan))
                  for t in (TYPE_SI, TYPE_O, TYPE_CA, TYPE_NA, TYPE_K)}
             for st in ("am", "iex")} for s in ("cao", "noca")}

for ix, (XX, (xna_cao, xka_cao, xt_cao, xna_noca, xka_noca, xt_noca)) in enumerate(sorted(XX_CONFIG.items())):
    print(f"XX={XX}%...")
    tags = {
        ("cao", "am"):   f"cao_am_x{xka_cao}_r{{r}}",
        ("cao", "iex"):  f"cao_iex1_xt{xt_cao}_r{{r}}",
        ("noca", "am"):  f"noca_am_x{xka_noca}_r{{r}}",
        ("noca", "iex"): f"noca_iex1_xt{xt_noca}_r{{r}}",
    }
    for (sys_, st), pat in tags.items():
        per_rep = [pressure_per_type(pat.format(r=r)) for r in REPLICAS]
        for t in (TYPE_SI, TYPE_O, TYPE_CA, TYPE_NA, TYPE_K):
            arrs = [pr[t] for pr in per_rep]
            if any(len(a) < 5 for a in arrs):
                continue
            stats[sys_][st][t][0][ix] = np.mean([a.mean() for a in arrs])
            stats[sys_][st][t][1][ix] = np.mean([fwhm(a) for a in arrs])

import sys
sys.path.insert(0, str(HERE.parent))
from paper_style import BOX, FS_LABEL, FS_TEXT, apply_rcparams, relabel
apply_rcparams()

# (tipo, nombre, color AM, color IEX): dos tonos por especie
MOD_SERIES = [
    (TYPE_NA, "Na", "#1a3a5c", "#4f86c6"),
    (TYPE_K,  "K",  "#2ca0c4", "#5bc3e0"),
    (TYPE_CA, "Ca", "#7b3f00", "#d9903f"),
]
SI_SERIES = [(TYPE_SI, "Si", "#8e44ad", "#b07cc9")]
O_SERIES = [(TYPE_O, "O", "#27ae60", "#5fd18d")]
OFFSET_PT = 11.5
XSHIFT_PT = 11.5
LABEL_FS = 13
TICK_FS = FS_TEXT
LABEL_AX_FS = FS_LABEL
TITLE_FS = FS_LABEL
LETTER_FS = FS_TEXT

# Geometria (pulgadas): cada "box" (a)-(d) es cuadrado de lado BOX;
# fila 1 (c,d) reparte ese mismo cuadrado en dos sub-paneles (O arriba, Si abajo).
# Lado real de las cajas de las figuras 2x2 (fig_elastic, fig_bo_nbo): ~3.9 in
PW = 3.9
PH = 3.9
LM, GAP_X, RM = 1.0, 0.6, 0.15
TM, GAP_Y, BM = 1.05, 0.55, 0.75
BREAK_GAP = 0.13
W = LM + 2 * PW + GAP_X + RM
H = TM + 2 * PH + GAP_Y + BM


def darker(color, f=0.72):
    r, g, b = to_rgb(color)
    return (r * f, g * f, b * f)


def add_axes_in(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def panel_origin(col, row):
    return LM + col * (PW + GAP_X), BM + (1 - row) * (PH + GAP_Y)


def draw_series(ax, sys_, series):
    items = []
    for t, name, c_am, c_iex in series:
        if sys_ == "noca" and t == TYPE_CA:
            continue
        for st, color, marker, ls, tag in (("am", c_am, "o", "-", "AM"), ("iex", c_iex, "s", "--", "IEX")):
            m, w = stats[sys_][st][t]
            ok = ~np.isnan(m)
            ax.plot(CHI[ok], m[ok], color=color, marker=marker, ls=ls, ms=6.5, lw=1.6,
                    label=relabel(f"{name} ({tag})"), zorder=3)
            for ix in np.where(ok)[0]:
                items.append((ix, m[ix], darker(color), w[ix], st))
    return items


def style(ax, ylim):
    ax.set_ylim(*ylim)
    ax.set_xlim(8, 112)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(labelsize=TICK_FS, which="both")
    ax.grid(lw=0.35, color="#dddddd", zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)


def place_labels(ax, items, ylim):
    """Numero del ancho pegado a cada punto, sobre o bajo el marcador (lado con mas espacio libre).
    Los de AM se corren ligeramente a la izquierda y los de IEX a la derecha para no pisarse."""
    ppp = 72.0 / ax.figure.dpi
    to_pts = lambda y: ax.transData.transform((0, y))[1] * ppp
    top, bot = to_pts(ylim[1]), to_pts(ylim[0])
    for ix in sorted({it[0] for it in items}):
        col = sorted([it for it in items if it[0] == ix and ylim[0] <= it[1] <= ylim[1]], key=lambda it: it[1])
        marks = [(0.0, to_pts(it[1])) for it in col]
        placed = []
        for k, (_, y, color, val, st) in enumerate(col):
            yp = to_pts(y)
            dx = -XSHIFT_PT if st == "am" else XSHIFT_PT
            others = [m for j, m in enumerate(marks) if j != k] + placed
            best = None
            for d in (+1, -1):
                pos = yp + d * OFFSET_PT
                if pos + 6 > top or pos - 6 < bot:
                    continue
                dist = min([np.hypot(dx - ox, pos - oy) for ox, oy in others] + [1e9])
                if best is None or dist > best[0]:
                    best = (dist, d, pos)
            if best is None:
                continue
            _, d, pos = best
            placed.append((dx, pos))
            ax.annotate(f"{val:.1f}", (CHI[ix], y), textcoords="offset points",
                        xytext=(dx, d * OFFSET_PT), ha="center", va="center", fontsize=LABEL_FS,
                        color=color, zorder=4)


def break_marks(ax, top_edge):
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=13, linestyle="none", color="k", mec="k",
              mew=1.0, clip_on=False)
    y = 1 if top_edge else 0
    ax.plot([0, 1], [y, y], transform=ax.transAxes, **kw)


fig = plt.figure(figsize=(W, H))
pending = []          # (ax, items, ylim) para colocar numeros al final
handles = {}

for col, (sys_, title) in enumerate((("cao", relabel("10CaO")), ("noca", "Ca-free"))):
    # --- fila 0: modificadores ---
    x0, y0 = panel_origin(col, 0)
    ax = add_axes_in(x0, y0, PW, PH)
    items = draw_series(ax, sys_, MOD_SERIES)
    style(ax, YLIM_MOD)
    ax.axhline(0, color="#aaaaaa", lw=0.8, ls=":", zorder=0)
    if col == 0:
        fig.text((x0 - 0.75) / W, (y0 + PH / 2) / H, r"$\langle P \rangle$ (GPa)", rotation=90, ha="center",
                 va="center", fontsize=LABEL_AX_FS)
    ax.text(0.5, 0.5, title, transform=ax.transAxes, ha="center", va="center", fontsize=TITLE_FS, zorder=5)
    ax.text(0.012, 0.975, "(a)" if col == 0 else "(b)", transform=ax.transAxes, ha="left", va="top",
            fontsize=LETTER_FS, fontweight="bold", zorder=5)
    pending.append((ax, items, YLIM_MOD))
    if col == 0:
        handles["mod"] = ax.get_legend_handles_labels()

    # --- fila 1: eje partido (O arriba, Si abajo) ---
    x1, y1 = panel_origin(col, 1)
    usable = PH - BREAK_GAP
    h_lo = usable * 30.0 / 50.0
    h_up = usable * 20.0 / 50.0
    ax_lo = add_axes_in(x1, y1, PW, h_lo)
    ax_up = add_axes_in(x1, y1 + h_lo + BREAK_GAP, PW, h_up)
    items_up = draw_series(ax_up, sys_, O_SERIES)
    items_lo = draw_series(ax_lo, sys_, SI_SERIES)
    style(ax_up, YLIM_O)
    style(ax_lo, YLIM_SI)
    ax_lo.set_yticks([-130, -120, -110])
    ax_up.spines["bottom"].set_visible(False)
    ax_up.tick_params(bottom=False, labelbottom=False, which="both")
    ax_lo.spines["top"].set_visible(False)
    ax_lo.set_xlabel(r"$\chi$ (%)", fontsize=LABEL_AX_FS)
    break_marks(ax_up, top_edge=False)
    break_marks(ax_lo, top_edge=True)
    if col == 0:
        fig.text((x1 - 0.75) / W, (y1 + PH / 2) / H, r"$\langle P \rangle$ (GPa)", rotation=90, ha="center",
                 va="center", fontsize=LABEL_AX_FS)
    fig.text((x1 + PW / 2) / W, (y1 + PH * 0.5) / H, title, ha="center", va="center", fontsize=TITLE_FS)
    ax_up.text(0.012, 0.955, "(c)" if col == 0 else "(d)", transform=ax_up.transAxes, ha="left", va="top",
               fontsize=LETTER_FS, fontweight="bold", zorder=5)
    pending.append((ax_up, items_up, YLIM_O))
    pending.append((ax_lo, items_lo, YLIM_SI))
    if col == 0:
        handles["o"] = ax_up.get_legend_handles_labels()
        handles["si"] = ax_lo.get_legend_handles_labels()

fig.canvas.draw()
for ax, items, yl in pending:
    place_labels(ax, items, yl)

hh = handles["mod"][0] + handles["si"][0] + handles["o"][0]
ll = handles["mod"][1] + handles["si"][1] + handles["o"][1]
fig.legend(hh, ll, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=5, fontsize=17,
           frameon=True, framealpha=0.9, edgecolor="#cccccc")

out = HERE / "fig_stress_av_mean_fwhm.pdf"
fig.savefig(out, dpi=300, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
print(f"Guardado: {out.name}  (paneles {PW:.2f} x {PH:.2f} in = 2:1)")
