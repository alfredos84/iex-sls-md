"""
fig_young_300K_ss.py
Módulo de Young E a 300 K — comparación entre:
  - Tensor elástico adiabático T=0 (STAGE8, diferencias finitas)
  - Curvas tensión-deformación a 300 K (STAGE9, erate=1e9 s⁻¹, ajuste lineal 0–2%)
PMMCS rc=8 Å — As-melted, IEX Proto1 y IEX Proto2.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
from pathlib import Path

HERE    = Path(__file__).parent
BASE    = HERE.parent
DIR_S8  = BASE / "STAGE8_ELASTIC_T0_300K" / "results"
DIR_S9  = BASE / "STAGE9_STRESS_STRAIN_300K" / "results"
OUT_DIR = HERE

XA_VALS = [0, 1, 3, 6, 9, 12, 15]
XT_VALS = [1, 3, 6, 9, 12, 15]
XP_VALS = [0, 1, 3, 6, 9, 12]

STRAIN_ELASTIC = 0.02   # rango de ajuste lineal [0, 2%]


# ── STAGE8: E del tensor elástico ────────────────────────────────────────────
def load_E_s8(tag):
    p = DIR_S8 / f"elastic_T0_Cij_PMMCS_rc8p0_{tag}.txt"
    if not p.exists():
        return np.nan
    lines = p.read_text().strip().split('\n')
    d = dict(zip(lines[0].split(), [float(v) for v in lines[1].split()]))
    return d['E']


def collect_E_s8(tags_fn, x_list):
    means, stds = [], []
    for x in x_list:
        vals = [load_E_s8(tags_fn(x, r)) for r in [1,2,3]]
        vals = [v for v in vals if not np.isnan(v)]
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals, ddof=1) if len(vals)>1 else 0.0)
    return np.array(means), np.array(stds)


E_am_s8,  dE_am_s8  = collect_E_s8(lambda x,r: f"AM300K_x{x}_r{r}",       XA_VALS)
E_ix1_s8, dE_ix1_s8 = collect_E_s8(lambda x,r: f"IEX1_300K_xt{x}_r{r}",   XT_VALS)
E_ix2_s8, dE_ix2_s8 = collect_E_s8(lambda x,r: f"IEX2_300K_xp{x}_r{r}",   XP_VALS)


# ── STAGE9: E de curvas tensión-deformación ───────────────────────────────────
def extract_E_ss(dat_file, strain_max=STRAIN_ELASTIC):
    if not Path(dat_file).exists():
        return np.nan
    try:
        data = np.loadtxt(dat_file, skiprows=1)
    except Exception:
        return np.nan
    if data.ndim < 2 or data.shape[0] < 3:
        return np.nan
    strain = data[:, 0]
    name   = Path(dat_file).name
    if '_x_' in name:
        stress = data[:, 5]
    elif '_y_' in name:
        stress = data[:, 6]
    else:
        stress = data[:, 7]
    mask = (strain >= 0.0) & (strain <= strain_max)
    if mask.sum() < 3:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        slope, _ = np.polyfit(strain[mask], stress[mask], 1)
    return slope


def collect_E_ss(tag_fn, x_list):
    means, stds = [], []
    for x in x_list:
        vals = []
        for r in [1,2,3]:
            for d in ['x','y','z']:
                tag = tag_fn(x, r)
                E = extract_E_ss(DIR_S9 / f"ss_{d}_{tag}.dat")
                if not np.isnan(E):
                    vals.append(E)
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals, ddof=1) if len(vals)>1 else 0.0)
    return np.array(means), np.array(stds)


E_am_ss,  dE_am_ss  = collect_E_ss(lambda x,r: f"AM300K_x{x}_r{r}",       XA_VALS)
E_ix1_ss, dE_ix1_ss = collect_E_ss(lambda x,r: f"IEX1_300K_xt{x}_r{r}",   XT_VALS)
E_ix2_ss, dE_ix2_ss = collect_E_ss(lambda x,r: f"IEX2_300K_xp{x}_r{r}",   XP_VALS)

# ── Tabla comparativa ─────────────────────────────────────────────────────────
print(f"\n{'x':>4}  {'E_T0 (GPa)':>11}  {'E_SS (GPa)':>11}  {'Δ (GPa)':>9}")
print("As-melted:")
for i,x in enumerate(XA_VALS):
    print(f"  {x:>2d}   {E_am_s8[i]:>11.2f}  {E_am_ss[i]:>11.2f}  {E_am_ss[i]-E_am_s8[i]:>9.2f}")
print("IEX Proto1:")
for i,x in enumerate(XT_VALS):
    print(f"  xt={x:>2d}  {E_ix1_s8[i]:>11.2f}  {E_ix1_ss[i]:>11.2f}  {E_ix1_ss[i]-E_ix1_s8[i]:>9.2f}")
print("IEX Proto2:")
for i,x in enumerate(XP_VALS):
    print(f"  xp={x:>2d}  {E_ix2_s8[i]:>11.2f}  {E_ix2_ss[i]:>11.2f}  {E_ix2_ss[i]-E_ix2_s8[i]:>9.2f}")

# ── Figura: 3 paneles (AM, IEX1, IEX2) — T0 vs SS ───────────────────────────
C_T0 = '#2c3e50'   # gris oscuro — tensor T=0K
C_SS = '#e67e22'   # naranja    — stress-strain 300K

KW_T0 = dict(lw=1.5, ms=7, capsize=3, capthick=1.0, elinewidth=0.8, zorder=4)
KW_SS = dict(lw=1.5, ms=7, capsize=3, capthick=1.0, elinewidth=0.8, zorder=3,
             ls='--')

configs = [
    ('As-melted',  XA_VALS, E_am_s8,  dE_am_s8,  E_am_ss,  dE_am_ss,  'o', r'$x$'),
    ('IEX Proto1', XT_VALS, E_ix1_s8, dE_ix1_s8, E_ix1_ss, dE_ix1_ss, 's', r'$x_\mathrm{target}$'),
    ('IEX Proto2', XP_VALS, E_ix2_s8, dE_ix2_s8, E_ix2_ss, dE_ix2_ss, 'D', r'$x_\mathrm{parent}$'),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)

for ax, (label, x_vals, E_t0, dE_t0, E_ss, dE_ss, marker, xlabel) in zip(axes, configs):
    ax.errorbar(x_vals, E_t0, yerr=dE_t0, color=C_T0, marker=marker,
                label=r'$T=0$ K (dif. finitas)', **KW_T0)
    ax.errorbar(x_vals, E_ss, yerr=dE_ss, color=C_SS, marker=marker,
                label=r'300 K ($\dot{\varepsilon}=10^9$ s$^{-1}$)', **KW_SS)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel(xlabel + r' (mol% K$_2$O)', fontsize=9.5)
    ax.set_xticks(x_vals)
    ax.set_xlim(-1, max(x_vals)+1)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.grid(axis='y', lw=0.4, color='#cccccc', zorder=0)
    ax.grid(axis='y', which='minor', lw=0.2, color='#eeeeee', zorder=0)
    ax.tick_params(labelsize=8.5)
    ax.legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor='#cccccc')

axes[0].set_ylabel(r"Young's modulus $E$ (GPa)", fontsize=10)

fig.suptitle(
    r'PMMCS $r_c=8$ Å — Módulo de Young a 300 K  |  '
    r'75SiO$_2\cdot$(15$-x$)Na$_2$O$\cdot x$K$_2$O$\cdot$10CaO',
    fontsize=10
)
fig.tight_layout()
fig.savefig(OUT_DIR / 'fig_young_300K_ss.pdf', dpi=300, bbox_inches='tight')
fig.savefig(OUT_DIR / 'fig_young_300K_ss.png', dpi=150, bbox_inches='tight')
print(f"\nGuardado: {OUT_DIR / 'fig_young_300K_ss.pdf'}")
plt.close(fig)
