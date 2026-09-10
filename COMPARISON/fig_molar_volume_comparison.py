"""
fig_molar_volume_comparison.py
Molar volume vs x — As-melted y IEX1, comparación CaO vs Ca-free

Sistemas:
  CaO   : 75SiO2·(15-x)Na2O·xK2O·10CaO  — x = 0,1,3,6,9,12,15
  Ca-free: 75SiO2·(25-x)Na2O·xK2O        — x = 0,1,5,10,15,20,25

Convenciones de figura:
  AS-melted CaO   : negro, cuadrado, línea sólida
  AS-melted noCa  : negro, triángulo, línea discontinua
  IEX1 CaO        : rojo,  cuadrado, línea sólida
  IEX1 noCa       : rojo,  triángulo, línea discontinua
  box on, relación de aspecto 1:1, Times New Roman 14 pt
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

HERE    = Path(__file__).parent
BASE    = HERE.parent

# ── Rutas ─────────────────────────────────────────────────────────────────────

CAO_AM_LOGS   = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE1_MELTQUENCH" / "logs"
CAO_AM_DATA   = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE1_MELTQUENCH" / "data" / "asmelted_723K"
CAO_IEX1_LOGS = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE3_IEX_PROTO1" / "logs"
CAO_IEX1_DATA = BASE / "75SiO2_15-xNa2O_xK2O_10CaO" / "STAGE3_IEX_PROTO1" / "data" / "iox1_723K"

NOCA_AM_LOGS   = BASE / "75SiO2_25-xNa2O_xK2O" / "STAGE1_MELTQUENCH" / "logs"
NOCA_AM_DATA   = BASE / "75SiO2_25-xNa2O_xK2O" / "STAGE1_MELTQUENCH" / "data" / "asmelted_723K"
NOCA_IEX1_LOGS = BASE / "75SiO2_25-xNa2O_xK2O" / "STAGE3_IEX_PROTO1" / "logs"
NOCA_IEX1_DATA = BASE / "75SiO2_25-xNa2O_xK2O" / "STAGE3_IEX_PROTO1" / "data" / "iox1_723K"

NA     = 6.02214076e23
N_MOLS = 5000

X_CAO_AM   = np.array([0, 1, 3, 6, 9, 12, 15])
X_CAO_IEX1 = np.array([1, 3, 6, 9, 12, 15])
X_NOCA_AM   = np.array([0, 1, 5, 10, 15, 20, 25])
X_NOCA_IEX1 = np.array([1, 5, 10, 15, 20, 25])

# Concentración relativa K2O/(Na2O+K2O): CaO total alkali=15, Ca-free=25
XREL_CAO_AM   = X_CAO_AM   / 15.0
XREL_CAO_IEX1 = X_CAO_IEX1 / 15.0
XREL_NOCA_AM   = X_NOCA_AM   / 25.0
XREL_NOCA_IEX1 = X_NOCA_IEX1 / 25.0


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


def load_series(x_vals, log_dir, data_dir, log_tmpl, data_tmpl, tail):
    reps = np.array([
        [vm_from_log(log_dir / log_tmpl.format(x=x, r=r),
                     data_dir / data_tmpl.format(x=x, r=r),
                     tail=tail)
         for r in (1, 2, 3)]
        for x in x_vals
    ])
    return reps.mean(axis=1), reps.std(axis=1, ddof=1)


# ── Cargar datos ──────────────────────────────────────────────────────────────

print("Cargando datos CaO — AM...")
Vm_cao_am_m, Vm_cao_am_s = load_series(
    X_CAO_AM, CAO_AM_LOGS, CAO_AM_DATA,
    "stage1_x{x}_r{r}_PMMCS_rc8p0.lammps",
    "AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data",
    tail=3)

print("Cargando datos CaO — IEX1...")
Vm_cao_iex1_m, Vm_cao_iex1_s = load_series(
    X_CAO_IEX1, CAO_IEX1_LOGS, CAO_IEX1_DATA,
    "stage3_xt{x}_r{r}_PMMCS_rc8p0.lammps",
    "IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data",
    tail=200)

print("Cargando datos Ca-free — AM...")
Vm_noca_am_m, Vm_noca_am_s = load_series(
    X_NOCA_AM, NOCA_AM_LOGS, NOCA_AM_DATA,
    "stage1_x{x}_r{r}_PMMCS_rc8p0.lammps",
    "AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data",
    tail=3)

print("Cargando datos Ca-free — IEX1...")
Vm_noca_iex1_m, Vm_noca_iex1_s = load_series(
    X_NOCA_IEX1, NOCA_IEX1_LOGS, NOCA_IEX1_DATA,
    "stage3_xt{x}_r{r}_PMMCS_rc8p0.lammps",
    "IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data",
    tail=200)

# ── Tabla resumen ─────────────────────────────────────────────────────────────

print(f"\n{'x':>3}  {'Vm_AM CaO':>14}  {'±':>7}  {'Vm_AM noCa':>14}  {'±':>7}")
for i, x in enumerate(X_CAO_AM):
    if x in X_NOCA_AM:
        j = list(X_NOCA_AM).index(x)
        print(f"  {x:>2d}   {Vm_cao_am_m[i]:>14.4f}  {Vm_cao_am_s[i]:>7.4f}"
              f"  {Vm_noca_am_m[j]:>14.4f}  {Vm_noca_am_s[j]:>7.4f}")
    else:
        print(f"  {x:>2d}   {Vm_cao_am_m[i]:>14.4f}  {Vm_cao_am_s[i]:>7.4f}  {'—':>14}")

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

fig, ax = plt.subplots(figsize=(6.5, 6.5))

# CaO (cuadrados, sólido)
ax.errorbar(XREL_CAO_AM,   Vm_cao_am_m,   yerr=Vm_cao_am_s,
            color=C_AM,   marker='s', ls='-',
            label=r'AM — 10CaO',  zorder=3, **KW)
ax.errorbar(XREL_CAO_IEX1, Vm_cao_iex1_m, yerr=Vm_cao_iex1_s,
            color=C_IEX1, marker='s', ls='-',
            label=r'IEX1 — 10CaO',  zorder=4, **KW)

# Ca-free (triángulos, discontinuo)
ax.errorbar(XREL_NOCA_AM,   Vm_noca_am_m,   yerr=Vm_noca_am_s,
            color=C_AM,   marker='^', ls='--',
            label=r'AM — Ca-free',   zorder=3, **KW)
ax.errorbar(XREL_NOCA_IEX1, Vm_noca_iex1_m, yerr=Vm_noca_iex1_s,
            color=C_IEX1_DARK, marker='^', ls='--',
            label=r'IEX1 — Ca-free', zorder=4, **KW)

ax.set_xlabel(r'K$_2$O / (Na$_2$O + K$_2$O) (%)', fontsize=14)
ax.set_ylabel(r'Molar Volume (cm$^3$ mol$^{-1}$)', fontsize=14)

ax.set_xlim(-0.03, 1.03)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v*100:.0f}'))
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

ax.tick_params(labelsize=12, which='both')

for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)

ax.grid(lw=0.35, color='#dddddd', zorder=0)

ax.legend(fontsize=11, frameon=True, framealpha=0.9,
          edgecolor='#cccccc', loc='best')

fig.tight_layout()
fig.savefig(HERE / 'fig_molar_volume_comparison.pdf', dpi=300, bbox_inches='tight')
fig.savefig(HERE / 'fig_molar_volume_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nGuardado: {HERE / 'fig_molar_volume_comparison.pdf'}")
plt.close(fig)
