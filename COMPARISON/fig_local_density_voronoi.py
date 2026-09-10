"""
fig_local_density_voronoi.py
Volumen de Voronoi V_Vor alrededor de Na y K.
Sin r_cut — volumen calculado con compute voronoi/atom de LAMMPS.
Genera fig_voronoi_XX{20,40,60,80,100}.pdf — 1 fila, 2 paneles: 10CaO | Ca-free.

Archivos de entrada (COMPARISON/VORONOI/):
  {tag}_vol.dat   — per-atom: id type vol nfaces
  {tag}_neigh.dat — per-face: id_i id_j area

Atom types: Si=1, O=2, Ca=3, Na=4, K=5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

HERE     = Path(__file__).parent
VORO_DIR = HERE / "VORONOI"

TYPE_O, TYPE_NA, TYPE_K = 2, 4, 5


# ── Lectura de dumps ──────────────────────────────────────────────────────────

def read_vol(path):
    data = {}
    in_atoms = False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("ITEM: ATOMS"):
                in_atoms = True; continue
            if s.startswith("ITEM:"):
                in_atoms = False; continue
            if in_atoms and s:
                parts = s.split()
                data[int(parts[0])] = (int(parts[1]), float(parts[2]))
    return data


def vol_for_type(vol_path, center_type):
    vol_data = read_vol(vol_path)
    return np.array([v for (t, v) in vol_data.values() if t == center_type])


def pool(tag_list, center_type):
    vols = []
    for tag in tag_list:
        vols.extend(vol_for_type(VORO_DIR / f"{tag}_vol.dat", center_type))
    return np.array(vols)


# ── Configuración por XX ──────────────────────────────────────────────────────

XX_CONFIG = {
    20:  (3,  3,  3,  5,  5,  5),
    40:  (6,  6,  6,  10, 10, 10),
    60:  (9,  9,  9,  15, 15, 15),
    80:  (12, 12, 12, 20, 20, 20),
    100: (0,  15, 15, 0,  25, 25),
}

# ── Figura ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':     'Times New Roman',
    'font.size':       17,
    'axes.linewidth':  0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

C_NA_AM = '#1a3a5c'
C_K_AM  = '#2ca0c4'
C_K_IEX = '#c0392b'
SERIES  = [
    ('Na (As-Melted)',    C_NA_AM, '-'),
    ('K (As-Melted)',     C_K_AM,  '--'),
    ('K (Ion-Exchanged)', C_K_IEX, ':'),
]


def norm_hist(data, bins):
    counts, edges = np.histogram(data, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    norm = counts / counts.max() if counts.max() > 0 else counts.astype(float)
    return centers, norm


def plot_nonzero(ax, cx, ny, **kw):
    mask = ny > 0
    ax.plot(cx[mask], ny[mask], **kw)


def decorate_vol(ax, xlim):
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(r'Voronoi Volume (Å$^3$)', fontsize=16)
    ax.set_ylabel('Normalized Amplitude', fontsize=16)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(labelsize=14, which='both')
    ax.grid(lw=0.35, color='#dddddd', zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8)


for XX, (xna_cao, xka_cao, xt_cao, xna_noca, xka_noca, xt_noca) in XX_CONFIG.items():
    print(f"\n── XX={XX}% ──")

    tags_na_cao  = [f"cao_am_x{xna_cao}_r{r}"    for r in (1, 2, 3)]
    tags_ka_cao  = [f"cao_am_x{xka_cao}_r{r}"    for r in (1, 2, 3)]
    tags_ki_cao  = [f"cao_iex1_xt{xt_cao}_r{r}"  for r in (1, 2, 3)]
    tags_na_noca = [f"noca_am_x{xna_noca}_r{r}"  for r in (1, 2, 3)]
    tags_ka_noca = [f"noca_am_x{xka_noca}_r{r}"  for r in (1, 2, 3)]
    tags_ki_noca = [f"noca_iex1_xt{xt_noca}_r{r}" for r in (1, 2, 3)]

    cao_v_na  = pool(tags_na_cao,  TYPE_NA)
    cao_v_ka  = pool(tags_ka_cao,  TYPE_K)
    cao_v_ki  = pool(tags_ki_cao,  TYPE_K)
    noca_v_na = pool(tags_na_noca, TYPE_NA)
    noca_v_ka = pool(tags_ka_noca, TYPE_K)
    noca_v_ki = pool(tags_ki_noca, TYPE_K)

    print(f"  {'':28s}  {'<V_Vor>':>8}")
    for lbl, v in [
        ("CaO — Na(AM)",      cao_v_na),
        ("CaO — K(AM)",       cao_v_ka),
        ("CaO — K(IEX1)",     cao_v_ki),
        ("Ca-free — Na(AM)",  noca_v_na),
        ("Ca-free — K(AM)",   noca_v_ka),
        ("Ca-free — K(IEX1)", noca_v_ki),
    ]:
        if len(v) > 0:
            print(f"    {lbl:26s}  {v.mean():8.3f} Å³")
        else:
            print(f"    {lbl:26s}  (vacío)")

    all_v    = np.concatenate([cao_v_na, cao_v_ka, cao_v_ki,
                                noca_v_na, noca_v_ka, noca_v_ki])
    vol_bins = np.linspace(all_v.min() * 0.9, all_v.max() * 1.05, 45)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.5))

    for col, (vna, vka, vki, title) in enumerate([
        (cao_v_na,  cao_v_ka,  cao_v_ki,  rf'10CaO — $\chi$ = {XX}%'),
        (noca_v_na, noca_v_ka, noca_v_ki, rf'Ca-free — $\chi$ = {XX}%'),
    ]):
        ax = axes[col]
        for data, (lbl, color, ls) in zip([vna, vka, vki], SERIES):
            if len(data) == 0:
                continue
            cx, ny = norm_hist(data, vol_bins)
            plot_nonzero(ax, cx, ny, color=color, ls=ls, lw=1.8,
                         marker='s', ms=4, label=lbl)
        ax.set_title(title, fontsize=16)
        decorate_vol(ax, (vol_bins[0], vol_bins[-1]))
        if col == 0:
            ax.legend(fontsize=13, frameon=True, framealpha=0.9,
                      edgecolor='#cccccc', loc='upper right')

    fig.tight_layout()
    out = HERE / f'fig_voronoi_XX{XX}.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"  Guardado: {out.name}")
    plt.close(fig)

print("\nListo.")
