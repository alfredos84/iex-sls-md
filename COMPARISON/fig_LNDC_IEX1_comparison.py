"""
fig_LNDC_IEX1_comparison.py
LNDC (Linear Network Dilation Coefficient) vs K2O/(Na2O+K2O) — comparación CaO vs Ca-free

Fórmula (Vargheese 2014):
  LNDC_AM   = slope(ln Vm_AM vs x_rel) / 3 * 1000    [ppk per unit x_rel]  → línea horizontal
  LNDC_IEX1 = (ln Vm_IEX(x_t) - ln Vm_AM(0)) / (3 * x_rel_t) * 1000

Ajuste logarítmico: LNDC ~ a*ln(x_rel) + b  sobre los puntos IEX1 de cada sistema.
Eje x: concentración relativa K2O/(Na2O+K2O), x_rel ∈ [0,1].
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

HERE    = Path(__file__).parent
BASE    = HERE.parent

CAO_AM_DATA   = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE1_MELTQUENCH" / "data" / "asmelted_723K"
CAO_AM_LOGS   = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE1_MELTQUENCH" / "logs"
CAO_IEX1_DATA = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE3_IEX_PROTO1" / "data" / "iox1_723K"
CAO_IEX1_LOGS = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE3_IEX_PROTO1" / "logs"

NOCA_AM_DATA   = BASE / "75SiO2_25-xNa2O_xK2O" / "STAGE1_MELTQUENCH" / "data" / "asmelted_723K"
NOCA_AM_LOGS   = BASE / "75SiO2_25-xNa2O_xK2O" / "STAGE1_MELTQUENCH" / "logs"
NOCA_IEX1_DATA = BASE / "75SiO2_25-xNa2O_xK2O" / "STAGE3_IEX_PROTO1" / "data" / "iox1_723K"
NOCA_IEX1_LOGS = BASE / "75SiO2_25-xNa2O_xK2O" / "STAGE3_IEX_PROTO1" / "logs"

NA     = 6.02214076e23
N_MOLS = 5000

X_CAO_AM    = np.array([0, 1, 3, 6, 9, 12, 15])
X_CAO_IEX1  = np.array([3, 6, 9, 12, 15])       # x=1 excluido (mismo criterio que Ca-free)
X_NOCA_AM   = np.array([0, 1, 5, 10, 15, 20, 25])
X_NOCA_IEX1 = np.array([5, 10, 15, 20, 25])     # x=1 excluido

XREL_CAO_AM   = X_CAO_AM   / 15.0
XREL_CAO_IEX1 = X_CAO_IEX1 / 15.0
XREL_NOCA_AM  = X_NOCA_AM  / 25.0
XREL_NOCA_IEX1= X_NOCA_IEX1 / 25.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def data_timestep(data_path):
    with open(data_path, encoding='utf-8', errors='replace') as fh:
        first = fh.readline()
    m = re.search(r'timestep\s*=\s*(\d+)', first)
    if m is None:
        raise ValueError(f"No timestep en cabecera: {data_path}")
    return int(m.group(1))


def log_vol_avg(log_path, cutoff_step, tail=200):
    vols = []
    with open(log_path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                step = int(parts[0])
                vol  = float(parts[6])
            except ValueError:
                continue
            if step <= cutoff_step:
                vols.append(vol)
    if not vols:
        raise RuntimeError(f"Sin datos thermo ≤ step {cutoff_step} en {log_path}")
    return float(np.mean(vols[-tail:]))


def vm_from_log(log_path, data_path, tail=200):
    cutoff = data_timestep(data_path)
    V_avg  = log_vol_avg(log_path, cutoff, tail=tail)
    return V_avg * NA * 1e-24 / N_MOLS


def load_vm_am_reps(x_vals, log_dir, data_dir, tail=3):
    """Devuelve array (n_x, 3) — Vm por réplica."""
    return np.array([
        [vm_from_log(
            log_dir  / f"stage1_x{x}_r{r}_PMMCS_rc8p0.lammps",
            data_dir / f"AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data",
            tail=tail) for r in (1, 2, 3)]
        for x in x_vals])


def load_vm_iex1_reps(x_vals, log_dir, data_dir, tail=200):
    """Devuelve array (n_x, 3) — Vm por réplica."""
    return np.array([
        [vm_from_log(
            log_dir  / f"stage3_xt{x}_r{r}_PMMCS_rc8p0.lammps",
            data_dir / f"IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data",
            tail=tail) for r in (1, 2, 3)]
        for x in x_vals])


# ── Cargar volúmenes molares (por réplica) ────────────────────────────────────

Vm_cao_am_r   = load_vm_am_reps(X_CAO_AM,   CAO_AM_LOGS,   CAO_AM_DATA)
Vm_cao_iex1_r = load_vm_iex1_reps(X_CAO_IEX1, CAO_IEX1_LOGS, CAO_IEX1_DATA)
Vm_noca_am_r  = load_vm_am_reps(X_NOCA_AM,  NOCA_AM_LOGS,  NOCA_AM_DATA)
Vm_noca_iex1_r= load_vm_iex1_reps(X_NOCA_IEX1, NOCA_IEX1_LOGS, NOCA_IEX1_DATA)

# Medias para los ajustes lineales del AM
Vm_cao_am   = Vm_cao_am_r.mean(axis=1)
Vm_noca_am  = Vm_noca_am_r.mean(axis=1)

# ── LNDC — calculado con x absoluto (mol%) para mantener unidades ppk/mol% ───

# AM: slope de ln(Vm) vs x_abs [mol%], /3 * 1000  →  ppk/mol%
slope_cao,  _ = np.polyfit(X_CAO_AM,  np.log(Vm_cao_am),  1)
slope_noca, _ = np.polyfit(X_NOCA_AM, np.log(Vm_noca_am), 1)
LNDC_cao_am  = slope_cao  / 3.0 * 1000   # ppk/mol%
LNDC_noca_am = slope_noca / 3.0 * 1000   # ppk/mol%

# IEX1 por réplica: LNDC_r = (ln Vm_IEX_r - ln Vm_AM0_mean) / (3*x_abs) * 1000
# Usamos Vm_AM(x=0) medio como referencia (x=0 es el índice 0 en X_CAO_AM / X_NOCA_AM)
Vm_cao_am0  = Vm_cao_am[0]
Vm_noca_am0 = Vm_noca_am[0]

LNDC_cao_iex1_r = np.array([
    (np.log(Vm_cao_iex1_r[i]) - np.log(Vm_cao_am0)) / (3.0 * X_CAO_IEX1[i]) * 1000
    for i in range(len(X_CAO_IEX1))])   # shape (n_x, 3)

LNDC_noca_iex1_r = np.array([
    (np.log(Vm_noca_iex1_r[i]) - np.log(Vm_noca_am0)) / (3.0 * X_NOCA_IEX1[i]) * 1000
    for i in range(len(X_NOCA_IEX1))])  # shape (n_x, 3)

LNDC_cao_iex1  = LNDC_cao_iex1_r.mean(axis=1)
LNDC_cao_err   = LNDC_cao_iex1_r.std(axis=1, ddof=1)
LNDC_noca_iex1 = LNDC_noca_iex1_r.mean(axis=1)
LNDC_noca_err  = LNDC_noca_iex1_r.std(axis=1, ddof=1)

print(f"LNDC AM — CaO    : {LNDC_cao_am:.4f} ppk/mol%")
print(f"LNDC AM — Ca-free: {LNDC_noca_am:.4f} ppk/mol%")
print(f"\n{'x_rel':>8}  {'LNDC IEX1 CaO':>16} {'±':>7}  {'LNDC IEX1 noCa':>16} {'±':>7}")
for i in range(len(X_CAO_IEX1)):
    j = i  # misma longitud
    print(f"  {XREL_CAO_IEX1[i]:.3f}    {LNDC_cao_iex1[i]:>16.4f} {LNDC_cao_err[i]:>7.4f}"
          f"  {LNDC_noca_iex1[j]:>16.4f} {LNDC_noca_err[j]:>7.4f}")

# ── Ajustes logarítmicos (sobre x_rel para trazar, coef en escala x_rel) ─────

a_cao,  b_cao  = np.polyfit(np.log(XREL_CAO_IEX1),  LNDC_cao_iex1,  1)
a_noca, b_noca = np.polyfit(np.log(XREL_NOCA_IEX1), LNDC_noca_iex1, 1)

xrel_fit = np.linspace(0.03, 1.0, 400)
fit_cao  = a_cao  * np.log(xrel_fit) + b_cao
fit_noca = a_noca * np.log(xrel_fit) + b_noca

# ── Figura ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':     'Times New Roman',
    'font.size':       14,
    'axes.linewidth':  0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

C_AM       = 'black'
C_IEX1     = '#e74c3c'   # rojo claro — CaO
C_IEX1_DARK= '#7b241c'   # rojo oscuro — Ca-free

fig, ax = plt.subplots(figsize=(6.5, 6.5))

# Líneas AM horizontales
ax.axhline(LNDC_cao_am,  color=C_AM, ls='-',  lw=1.8, label='AM — 10CaO',   zorder=2)
ax.axhline(LNDC_noca_am, color=C_AM, ls='--', lw=1.8, label='AM — Ca-free', zorder=2)

# Puntos IEX1 con barras de error
KW = dict(ms=7, lw=0, capsize=3, capthick=1.0, elinewidth=0.8)
ax.errorbar(XREL_CAO_IEX1,  LNDC_cao_iex1,  yerr=LNDC_cao_err,
            marker='s', color=C_IEX1,      label='IEX1 — 10CaO',   zorder=4, **KW)
ax.errorbar(XREL_NOCA_IEX1, LNDC_noca_iex1, yerr=LNDC_noca_err,
            marker='^', color=C_IEX1_DARK, label='IEX1 — Ca-free',  zorder=4, **KW)

# Ajustes logarítmicos (sin entrada en leyenda)
ax.plot(xrel_fit, fit_cao,  color=C_IEX1,      lw=1.2, ls='-',  zorder=3, label='_nolegend_')
ax.plot(xrel_fit, fit_noca, color=C_IEX1_DARK, lw=1.2, ls='--', zorder=3, label='_nolegend_')

ax.set_xlabel(r'K$_2$O / (Na$_2$O + K$_2$O) (%)', fontsize=14)
ax.set_ylabel(r'LNDC (ppk mol%$^{-1}$)', fontsize=14)

ax.set_xlim(-0.03, 1.03)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v*100:.0f}'))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.set_ylim(bottom=0)

ax.tick_params(labelsize=12, which='both')
ax.grid(lw=0.35, color='#dddddd', zorder=0)
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)

ax.legend(fontsize=11, frameon=True, framealpha=0.9,
          edgecolor='#cccccc', loc='lower right')

# Eje secundario: % del LNDC AM — referencia = CaO AM (línea sólida en 100%)
LNDC_am_ref = LNDC_cao_am
ymax = ax.get_ylim()[1]
ax2 = ax.twinx()
ax2.set_ylim(0, ymax / LNDC_am_ref * 100.0)
pct_ticks = [0, 20, 40, 60, 80, 100]
ax2.set_yticks(pct_ticks)
ax2.set_yticklabels([f'{p}%' for p in pct_ticks], fontsize=12)
ax2.set_ylabel('% of As-melted LNDC', fontsize=14)
ax2.tick_params(labelsize=12)

fig.tight_layout()
fig.savefig(HERE / 'fig_LNDC_IEX1_comparison.pdf', dpi=300, bbox_inches='tight')
fig.savefig(HERE / 'fig_LNDC_IEX1_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nGuardado: {HERE / 'fig_LNDC_IEX1_comparison.pdf'}")
plt.close(fig)
