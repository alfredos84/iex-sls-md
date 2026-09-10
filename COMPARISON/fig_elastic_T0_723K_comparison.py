"""
fig_elastic_T0_723K_comparison.py
Propiedades elásticas (E, K, G, nu) vs concentración relativa K2O/(Na2O+K2O)
Comparación CaO vs Ca-free — As-melted y IEX1, 723 K, PMMCS rc=8 Å

Cij file columns: C11..C56 KV GV K G E nu
Índices (0-based): 23=K, 24=G, 25=E, 26=nu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

HERE    = Path(__file__).parent
BASE    = HERE.parent

CAO_AM_RES   = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE2_ELASTIC_T0"   / "results"
CAO_IEX1_RES = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE5_ELASTIC_T0_PROTO1" / "results"
NOCA_AM_RES   = BASE / "75SiO2_25-xNa2O_xK2O" / "STAGE2_ELASTIC_T0"   / "results"
NOCA_IEX1_RES = BASE / "75SiO2_25-xNa2O_xK2O" / "STAGE5_ELASTIC_T0_PROTO1" / "results"

X_CAO_AM   = np.array([0, 1, 3, 6, 9, 12, 15])
X_CAO_IEX1 = np.array([1, 3, 6, 9, 12, 15])
X_NOCA_AM   = np.array([0, 1, 5, 10, 15, 20, 25])
X_NOCA_IEX1 = np.array([1, 5, 10, 15, 20, 25])

XREL_CAO_AM   = X_CAO_AM   / 15.0
XREL_CAO_IEX1 = X_CAO_IEX1 / 15.0
XREL_NOCA_AM   = X_NOCA_AM   / 25.0
XREL_NOCA_IEX1 = X_NOCA_IEX1 / 25.0

# Column indices in Cij file (0-based, after header)
I_K, I_G, I_E, I_NU = 23, 24, 25, 26


def read_cij(path):
    data = np.loadtxt(path, skiprows=1)
    return data  # shape (27,)


def load_am(x_vals, res_dir, tmpl):
    """Returns (n_x, 4) array: K, G, E, nu — mean; and (n_x, 4) std."""
    means, stds = [], []
    for x in x_vals:
        reps = []
        for r in (1, 2, 3):
            p = res_dir / tmpl.format(x=x, r=r)
            d = read_cij(p)
            reps.append([d[I_K], d[I_G], d[I_E], d[I_NU]])
        reps = np.array(reps)
        means.append(reps.mean(axis=0))
        stds.append(reps.std(axis=0, ddof=1))
    return np.array(means), np.array(stds)


def load_iex1(x_vals, res_dir, tmpl):
    means, stds = [], []
    for x in x_vals:
        reps = []
        for r in (1, 2, 3):
            p = res_dir / tmpl.format(x=x, r=r)
            d = read_cij(p)
            reps.append([d[I_K], d[I_G], d[I_E], d[I_NU]])
        reps = np.array(reps)
        means.append(reps.mean(axis=0))
        stds.append(reps.std(axis=0, ddof=1))
    return np.array(means), np.array(stds)


# ── Cargar datos ──────────────────────────────────────────────────────────────

cao_am_m,   cao_am_s   = load_am(X_CAO_AM,   CAO_AM_RES,
    "elastic_T0_Cij_PMMCS_rc8p0_x{x}_r{r}.txt")
cao_iex1_m, cao_iex1_s = load_iex1(X_CAO_IEX1, CAO_IEX1_RES,
    "elastic_T0_Cij_PMMCS_rc8p0_IEX1_xt{x}_r{r}.txt")
noca_am_m,   noca_am_s   = load_am(X_NOCA_AM,   NOCA_AM_RES,
    "elastic_T0_Cij_PMMCS_rc8p0_x{x}_r{r}.txt")
noca_iex1_m, noca_iex1_s = load_iex1(X_NOCA_IEX1, NOCA_IEX1_RES,
    "elastic_T0_Cij_PMMCS_rc8p0_IEX1_xt{x}_r{r}.txt")

# ── Figura ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':     'Times New Roman',
    'font.size':       14,
    'axes.linewidth':  0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

C_AM        = 'black'
C_IEX1      = '#e74c3c'   # rojo claro — CaO
C_IEX1_DARK = '#7b241c'   # rojo oscuro — Ca-free
KW = dict(ms=7, lw=1.5, capsize=3, capthick=1.0, elinewidth=0.8)

props   = [2, 0, 1, 3]          # E, K, G, nu  (índices en columnas cargadas)
ylabels = [r'$E$ (GPa)', r'$K$ (GPa)', r'$G$ (GPa)', r'$\nu$']

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
axes = axes.flatten()

for ax, pi, ylabel in zip(axes, props, ylabels):

    ax.errorbar(XREL_CAO_AM,   cao_am_m[:,pi],   yerr=cao_am_s[:,pi],
                color=C_AM,   marker='s', ls='-',  label='AM — 10CaO',   zorder=3, **KW)
    ax.errorbar(XREL_CAO_IEX1, cao_iex1_m[:,pi], yerr=cao_iex1_s[:,pi],
                color=C_IEX1,      marker='s', ls='-',  label='IEX1 — 10CaO',   zorder=4, **KW)
    ax.errorbar(XREL_NOCA_AM,   noca_am_m[:,pi],   yerr=noca_am_s[:,pi],
                color=C_AM,        marker='^', ls='--', label='AM — Ca-free',    zorder=3, **KW)
    ax.errorbar(XREL_NOCA_IEX1, noca_iex1_m[:,pi], yerr=noca_iex1_s[:,pi],
                color=C_IEX1_DARK, marker='^', ls='--', label='IEX1 — Ca-free', zorder=4, **KW)

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v*100:.0f}'))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(labelsize=12, which='both')
    ax.grid(lw=0.35, color='#dddddd', zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)

# x label solo en fila inferior
for ax in axes[2:]:
    ax.set_xlabel(r'K$_2$O / (Na$_2$O + K$_2$O) (%)', fontsize=14)

fig.tight_layout(rect=[0, 0, 1, 0.93])

# Leyenda horizontal compartida, encima de los 4 paneles
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='upper center', ncol=4,
           fontsize=11, frameon=True, framealpha=0.9,
           edgecolor='#cccccc',
           bbox_to_anchor=(0.5, 0.97))
fig.savefig(HERE / 'fig_elastic_T0_723K_comparison.pdf', dpi=300, bbox_inches='tight')
fig.savefig(HERE / 'fig_elastic_T0_723K_comparison.png', dpi=150, bbox_inches='tight')
print(f"Guardado: {HERE / 'fig_elastic_T0_723K_comparison.pdf'}")
plt.close(fig)
