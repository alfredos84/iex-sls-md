"""
fig_bo_nbo_batch.py
Distribuciones de CN_BO y CN_NBO para Na, K (y Ca en vidrios CaO) — AM y IEX1.

Clasificación de oxígenos (r_cut Si–O = 2.0 Å):
  BO  : O con exactamente 2 vecinos Si
  NBO : O con exactamente 1 vecino Si

Radios de corte por especie:
  Si–O : RCUT_SIO = 2.00 Å  (primer mínimo g(r) Si-O — valle limpio)
  Na–O : RCUT_NA  = 3.21 Å
  K–O  : RCUT_K   = 3.77 Å  (AM); RCUT_IEX = 3.49 Å (IEX1, promedio Na/K)
  Ca–O : RCUT_CA  = 3.20 Å  (valle amplio 2.6–3.5 Å, segunda capa desde ~3.5 Å)

Genera fig_bo_nbo_XX{20,40,60,80,100}.pdf — 2 filas × 2 columnas:
  fila 0 (top):    CN_BO  — izquierda: 10CaO | derecha: Ca-free
  fila 1 (bottom): CN_NBO — izquierda: 10CaO | derecha: Ca-free

Atom types: Si=1, O=2, Ca=3, Na=4, K=5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.spatial import cKDTree

HERE  = Path(__file__).parent
BASE  = HERE.parent.parent

CAO_AM_DIR    = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE1_MELTQUENCH"  / "data" / "asmelted_723K"
CAO_IEX1_DIR  = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE3_IEX_PROTO1"  / "data" / "iox1_723K"
NOCA_AM_DIR   = BASE / "75SiO2_25-xNa2O_xK2O"        / "STAGE1_MELTQUENCH"  / "data" / "asmelted_723K"
NOCA_IEX1_DIR = BASE / "75SiO2_25-xNa2O_xK2O"        / "STAGE3_IEX_PROTO1"  / "data" / "iox1_723K"

RCUT_SIO = 2.00
RCUT_NA  = 3.21
RCUT_K   = 3.77
RCUT_IEX = (RCUT_NA + RCUT_K) / 2.0   # 3.49 Å
RCUT_CA  = 3.20

TYPE_SI, TYPE_O, TYPE_CA, TYPE_NA, TYPE_K = 1, 2, 3, 4, 5

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


# ── Clasificación BO / NBO ────────────────────────────────────────────────────

def classify_oxygens(pos, types, box):
    """
    Clasifica cada átomo O según sus vecinos Si a r < RCUT_SIO.
    Retorna dos arrays booleanos de longitud len(pos):
      is_bo[i]  = True si átomo i es O con 2 vecinos Si  (BO)
      is_nbo[i] = True si átomo i es O con 1 vecino Si   (NBO)
    """
    tree = cKDTree(pos, boxsize=box)
    o_mask = (types == TYPE_O)
    o_idx  = np.where(o_mask)[0]

    si_count = np.zeros(len(pos), dtype=int)
    for idx in o_idx:
        nbrs = tree.query_ball_point(pos[idx], RCUT_SIO)
        si_count[idx] = int(np.sum(types[nbrs] == TYPE_SI))

    is_bo  = o_mask & (si_count == 2)
    is_nbo = o_mask & (si_count == 1)
    return is_bo, is_nbo


def cn_bo_nbo_around(pos, types, box, center_type, rcut, is_bo, is_nbo):
    """
    Para cada átomo del tipo center_type, cuenta cuántos vecinos BO y NBO
    hay dentro de rcut.
    Retorna (cn_bo_arr, cn_nbo_arr), un valor por átomo central.
    """
    tree = cKDTree(pos, boxsize=box)
    centers = pos[types == center_type]
    cn_bo_list, cn_nbo_list = [], []
    for cp in centers:
        nbrs = np.array(tree.query_ball_point(cp, rcut), dtype=int)
        cn_bo_list.append(int(np.sum(is_bo[nbrs])))
        cn_nbo_list.append(int(np.sum(is_nbo[nbrs])))
    return np.array(cn_bo_list), np.array(cn_nbo_list)


# ── Cargadores por sistema ────────────────────────────────────────────────────

def load_am(am_dir, x, ctype, rcut):
    bo_all, nbo_all = [], []
    for r in (1, 2, 3):
        pos, types, box = read_data(am_dir / f"AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data")
        is_bo, is_nbo  = classify_oxygens(pos, types, box)
        cb, cn = cn_bo_nbo_around(pos, types, box, ctype, rcut, is_bo, is_nbo)
        bo_all.extend(cb); nbo_all.extend(cn)
    return np.array(bo_all), np.array(nbo_all)


def load_iex1(iex_dir, x, ctype, rcut):
    bo_all, nbo_all = [], []
    for r in (1, 2, 3):
        pos, types, box = read_data(iex_dir / f"IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data")
        is_bo, is_nbo  = classify_oxygens(pos, types, box)
        cb, cn = cn_bo_nbo_around(pos, types, box, ctype, rcut, is_bo, is_nbo)
        bo_all.extend(cb); nbo_all.extend(cn)
    return np.array(bo_all), np.array(nbo_all)


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
C_CA_AM = '#7b3f00'   # marrón — Ca en AM


def norm_hist(data, bins):
    counts, edges = np.histogram(data, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    norm = counts / counts.max() if counts.max() > 0 else counts.astype(float)
    return centers, norm


def plot_nonzero(ax, cx, ny, **kw):
    mask = ny > 0
    ax.plot(cx[mask], ny[mask], **kw)


def decorate(ax, xlabel, xlim):
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(xlabel, fontsize=16)
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
         xna_noca, xka_noca, xt_noca) in sorted(XX_CONFIG.items()):

    print(f"\n── XX={XX}% ──")

    # CaO — Na, K(AM), K(IEX1), Ca(AM)
    cao_na_bo,  cao_na_nbo  = load_am(CAO_AM_DIR,   xna_cao,  TYPE_NA,  RCUT_NA)
    cao_ka_bo,  cao_ka_nbo  = load_am(CAO_AM_DIR,   xka_cao,  TYPE_K,   RCUT_K)
    cao_ki_bo,  cao_ki_nbo  = load_iex1(CAO_IEX1_DIR, xt_cao, TYPE_K,   RCUT_IEX)

    if xka_cao > 0:
        cao_ca_bo,  cao_ca_nbo = load_am(CAO_AM_DIR, xka_cao, TYPE_CA, RCUT_CA)
    else:
        cao_ca_bo, cao_ca_nbo = np.array([0]), np.array([0])  # χ=100: solo K

    # Ca-free — Na, K(AM), K(IEX1)
    noca_na_bo, noca_na_nbo = load_am(NOCA_AM_DIR,   xna_noca, TYPE_NA, RCUT_NA)
    noca_ka_bo, noca_ka_nbo = load_am(NOCA_AM_DIR,   xka_noca, TYPE_K,  RCUT_K)
    noca_ki_bo, noca_ki_nbo = load_iex1(NOCA_IEX1_DIR, xt_noca, TYPE_K, RCUT_IEX)

    # Imprimir medias
    print(f"  {'Serie':28s}  {'<CN_BO>':>7}  {'<CN_NBO>':>8}")
    for lbl, bo, nbo in [
        ("CaO — Na(AM)",      cao_na_bo,  cao_na_nbo),
        ("CaO — K(AM)",       cao_ka_bo,  cao_ka_nbo),
        ("CaO — K(IEX1)",     cao_ki_bo,  cao_ki_nbo),
        ("CaO — Ca(AM)",      cao_ca_bo,  cao_ca_nbo),
        ("Ca-free — Na(AM)",  noca_na_bo, noca_na_nbo),
        ("Ca-free — K(AM)",   noca_ka_bo, noca_ka_nbo),
        ("Ca-free — K(IEX1)", noca_ki_bo, noca_ki_nbo),
    ]:
        print(f"    {lbl:26s}  {bo.mean():7.2f}  {nbo.mean():8.2f}")

    # Rango global de CN para bins comunes
    all_bo  = np.concatenate([cao_na_bo,  cao_ka_bo,  cao_ki_bo,
                               noca_na_bo, noca_ka_bo, noca_ki_bo])
    all_nbo = np.concatenate([cao_na_nbo, cao_ka_nbo, cao_ki_nbo,
                               noca_na_nbo, noca_ka_nbo, noca_ki_nbo])
    bo_max  = int(all_bo.max())  + 1
    nbo_max = int(all_nbo.max()) + 1
    bo_bins  = np.arange(-0.5, bo_max  + 1.5, 1.0)
    nbo_bins = np.arange(-0.5, nbo_max + 1.5, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 13))

    for col, (na_bo, ka_bo, ki_bo, ca_bo,
               na_nbo, ka_nbo, ki_nbo, ca_nbo,
               title, has_ca) in enumerate([
        (cao_na_bo,  cao_ka_bo,  cao_ki_bo,  cao_ca_bo,
         cao_na_nbo, cao_ka_nbo, cao_ki_nbo, cao_ca_nbo,
         rf'10CaO — $\chi$ = {XX}%', True),
        (noca_na_bo, noca_ka_bo, noca_ki_bo, None,
         noca_na_nbo, noca_ka_nbo, noca_ki_nbo, None,
         rf'Ca-free — $\chi$ = {XX}%', False),
    ]):
        # Fila 0: CN_BO
        ax0 = axes[0, col]
        cx, ny = norm_hist(na_bo, bo_bins)
        plot_nonzero(ax0, cx, ny, color=C_NA_AM, ls='-',  lw=1.8, marker='s', ms=4, label='Na (As-Melted)')
        cx, ny = norm_hist(ka_bo, bo_bins)
        plot_nonzero(ax0, cx, ny, color=C_K_AM,  ls='--', lw=1.8, marker='s', ms=4, label='K (As-Melted)')
        cx, ny = norm_hist(ki_bo, bo_bins)
        plot_nonzero(ax0, cx, ny, color=C_K_IEX, ls=':',  lw=1.8, marker='s', ms=4, label='K (Ion-Exchanged)')
        if has_ca:
            cx, ny = norm_hist(ca_bo, bo_bins)
            plot_nonzero(ax0, cx, ny, color=C_CA_AM, ls='-.', lw=1.8, marker='D', ms=4, label='Ca (As-Melted)')
        ax0.set_title(title, fontsize=16)
        decorate(ax0, r'CN$_\mathrm{BO}$', (-0.5, 15.5))
        ax0.legend(fontsize=13, frameon=True, framealpha=0.9, edgecolor='#cccccc', loc='upper right')

        # Fila 1: CN_NBO
        ax1 = axes[1, col]
        cx, ny = norm_hist(na_nbo, nbo_bins)
        plot_nonzero(ax1, cx, ny, color=C_NA_AM, ls='-',  lw=1.8, marker='s', ms=4, label='Na (As-Melted)')
        cx, ny = norm_hist(ka_nbo, nbo_bins)
        plot_nonzero(ax1, cx, ny, color=C_K_AM,  ls='--', lw=1.8, marker='s', ms=4, label='K (As-Melted)')
        cx, ny = norm_hist(ki_nbo, nbo_bins)
        plot_nonzero(ax1, cx, ny, color=C_K_IEX, ls=':',  lw=1.8, marker='s', ms=4, label='K (Ion-Exchanged)')
        if has_ca:
            cx, ny = norm_hist(ca_nbo, nbo_bins)
            plot_nonzero(ax1, cx, ny, color=C_CA_AM, ls='-.', lw=1.8, marker='D', ms=4, label='Ca (As-Melted)')
        decorate(ax1, r'CN$_\mathrm{NBO}$', (-0.5, 10.5))

    # Etiquetas de fila a la izquierda
    axes[0, 0].set_ylabel(r'CN$_\mathrm{BO}$ — Normalized Amplitude', fontsize=16)
    axes[1, 0].set_ylabel(r'CN$_\mathrm{NBO}$ — Normalized Amplitude', fontsize=16)
    for ax in axes[:, 1]:
        ax.set_ylabel('')

    fig.tight_layout()
    out = HERE / f'fig_bo_nbo_XX{XX}.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"  Guardado: {out.name}")
    plt.close(fig)

print("\nListo.")
