"""
fig_cn_av_batch.py
CN_O con r_cut promedio para K(IEX1): r_cut_iex = (RCUT_NA + RCUT_K) / 2.
Hipótesis: el K intercambiado ocupa un sitio de Na, por lo que su g(r) K-O
tiene un pico intermedio entre Na-O y K-O as-melted (Tandia 2012, Fig. 3).

  Na(AM)  : RCUT_NA = 3.21 Å
  K(AM)   : RCUT_K  = 3.77 Å
  K(IEX1) : RCUT_IEX = (3.21 + 3.77) / 2 = 3.49 Å

Genera fig_cn_av_XX{20,40,60,80,100}.pdf — 1 fila, 2 paneles: 10CaO | Ca-free.
Atom types: Si=1, O=2, Ca=3, Na=4, K=5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.spatial import cKDTree

HERE  = Path(__file__).parent
BASE  = HERE.parent

CAO_AM_DIR    = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE1_MELTQUENCH"  / "data" / "asmelted_723K"
CAO_IEX1_DIR  = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE3_IEX_PROTO1"  / "data" / "iox1_723K"
NOCA_AM_DIR   = BASE / "75SiO2_25-xNa2O_xK2O"        / "STAGE1_MELTQUENCH"  / "data" / "asmelted_723K"
NOCA_IEX1_DIR = BASE / "75SiO2_25-xNa2O_xK2O"        / "STAGE3_IEX_PROTO1"  / "data" / "iox1_723K"

RCUT_NA  = 3.21
RCUT_K   = 3.77
RCUT_IEX = (RCUT_NA + RCUT_K) / 2.0   # 3.49 Å
TYPE_O, TYPE_NA, TYPE_K = 2, 4, 5

print(f"RCUT_NA={RCUT_NA} Å  RCUT_K={RCUT_K} Å  RCUT_IEX={RCUT_IEX:.3f} Å")

XX_CONFIG = {
    20:  (3,  3,  3,  5,  5,  5),
    40:  (6,  6,  6,  10, 10, 10),
    60:  (9,  9,  9,  15, 15, 15),
    80:  (12, 12, 12, 20, 20, 20),
    100: (0,  15, 15, 0,  25, 25),
}


# ── I/O ───────────────────────────────────────────────────────────────────────

def read_data(path):
    box, lo = np.zeros(3), np.zeros(3)
    atoms, section = [], None
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if 'xlo xhi' in s:
                lo[0], hi = float(s.split()[0]), float(s.split()[1])
                box[0] = hi - lo[0]; continue
            elif 'ylo yhi' in s:
                lo[1], hi = float(s.split()[0]), float(s.split()[1])
                box[1] = hi - lo[1]; continue
            elif 'zlo zhi' in s:
                lo[2], hi = float(s.split()[0]), float(s.split()[1])
                box[2] = hi - lo[2]; continue
            if s[0].isalpha():
                section = 'atoms' if s.startswith('Atoms') else None
                continue
            if section == 'atoms':
                p = s.split()
                atoms.append((int(p[1]), float(p[3]), float(p[4]), float(p[5])))
    atoms = np.array(atoms)
    return atoms[:, 1:4] - lo, atoms[:, 0].astype(int), box


def cn_around(pos, types, box, center_type, rcut):
    tree = cKDTree(pos, boxsize=box)
    centers = pos[types == center_type]
    cn_list = []
    for neighbors in tree.query_ball_point(centers, rcut):
        cn_list.append(int(np.sum(types[neighbors] == TYPE_O)))
    return np.array(cn_list)


def load_am(am_dir, x, ctype):
    rcut = RCUT_NA if ctype == TYPE_NA else RCUT_K
    cn_all = []
    for r in (1, 2, 3):
        pos, types, box = read_data(
            am_dir / f"AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data")
        cn_all.extend(cn_around(pos, types, box, ctype, rcut))
    return np.array(cn_all)


def load_iex1(iex_dir, x):
    cn_all = []
    for r in (1, 2, 3):
        pos, types, box = read_data(
            iex_dir / f"IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data")
        cn_all.extend(cn_around(pos, types, box, TYPE_K, RCUT_IEX))
    return np.array(cn_all)


# ── Plot helpers ──────────────────────────────────────────────────────────────

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


def decorate_cn(ax, xlim):
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(r'O Coordination Number', fontsize=16)
    ax.set_ylabel('Normalized Amplitude', fontsize=16)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(labelsize=14, which='both')
    ax.grid(lw=0.35, color='#dddddd', zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8)


# ── Main loop ─────────────────────────────────────────────────────────────────

for XX, (xna_cao, xka_cao, xt_cao,
         xna_noca, xka_noca, xt_noca) in XX_CONFIG.items():

    print(f"\n── XX={XX}% ──")

    cao_cn_na  = load_am(CAO_AM_DIR,      xna_cao,  TYPE_NA)
    cao_cn_ka  = load_am(CAO_AM_DIR,      xka_cao,  TYPE_K)
    cao_cn_ki  = load_iex1(CAO_IEX1_DIR,  xt_cao)
    noca_cn_na = load_am(NOCA_AM_DIR,     xna_noca, TYPE_NA)
    noca_cn_ka = load_am(NOCA_AM_DIR,     xka_noca, TYPE_K)
    noca_cn_ki = load_iex1(NOCA_IEX1_DIR, xt_noca)

    print(f"  {'':28s}  {'<CN_O>':>7}")
    for lbl, cn in [
        ("CaO — Na(AM)",      cao_cn_na),
        ("CaO — K(AM)",       cao_cn_ka),
        ("CaO — K(IEX1)",     cao_cn_ki),
        ("Ca-free — Na(AM)",  noca_cn_na),
        ("Ca-free — K(AM)",   noca_cn_ka),
        ("Ca-free — K(IEX1)", noca_cn_ki),
    ]:
        print(f"    {lbl:26s}  {cn.mean():.2f}")

    all_cn  = np.concatenate([cao_cn_na, cao_cn_ka, cao_cn_ki,
                               noca_cn_na, noca_cn_ka, noca_cn_ki])
    CN_MAX  = int(all_cn.max()) + 1
    cn_bins = np.arange(-0.5, CN_MAX + 1.5, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.5))

    for col, (cn_na, cn_ka, cn_ki, title) in enumerate([
        (cao_cn_na,  cao_cn_ka,  cao_cn_ki,  rf'10CaO — $\chi$ = {XX}%'),
        (noca_cn_na, noca_cn_ka, noca_cn_ki, rf'Ca-free — $\chi$ = {XX}%'),
    ]):
        ax = axes[col]
        for data, (lbl, color, ls) in zip([cn_na, cn_ka, cn_ki], SERIES):
            cx, ny = norm_hist(data, cn_bins)
            plot_nonzero(ax, cx, ny, color=color, ls=ls, lw=1.8,
                         marker='s', ms=4, label=lbl)
        ax.set_title(title, fontsize=16)
        decorate_cn(ax, (-0.5, CN_MAX + 0.5))
        if col == 0:
            ax.legend(fontsize=13, frameon=True, framealpha=0.9,
                      edgecolor='#cccccc', loc='upper right')

    fig.tight_layout()
    out = HERE / f'fig_cn_av_XX{XX}.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"  Guardado: {out.name}")
    plt.close(fig)

print("\nListo.")
