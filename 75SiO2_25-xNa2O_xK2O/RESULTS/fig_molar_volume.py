"""
fig_molar_volume.py
Molar volume vs x — As-melted, IEX1 y IEX2
Sistema: 75SiO2·(25-x)Na2O·xK2O·10CaO (mol%), PMMCS rc=8 Å, 723 K

As-melted : x          = 0, 1, 3, 6, 9, 12, 15  (tail=3 : últimas 3 entradas del quench)
IEX Proto1: x_target   = 1, 3, 6, 9, 12, 15      (tail=200: NPT 723 K)
IEX Proto2: x_parent   = 0, 1, 3, 6, 9, 12       (tail=200: NPT 723 K tras reemplazo total Na→K)
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

HERE      = Path(__file__).parent
BASE      = HERE.parent

AM_DATA   = BASE / "STAGE1_MELTQUENCH" / "data" / "asmelted_723K"
AM_LOGS   = BASE / "STAGE1_MELTQUENCH" / "logs"
IEX1_DATA = BASE / "STAGE3_IEX_PROTO1" / "data" / "iox1_723K"
IEX1_LOGS = BASE / "STAGE3_IEX_PROTO1" / "logs"
IEX2_DATA = BASE / "STAGE4_IEX_PROTO2" / "data" / "iox2_723K"
IEX2_LOGS = BASE / "STAGE4_IEX_PROTO2" / "logs"

NA     = 6.02214076e23
N_MOLS = 5000

X_ALL  = np.array([0, 1, 3, 6, 9, 12, 15])
X_IEX1 = np.array([1, 3, 6, 9, 12, 15])
X_IEX2 = np.array([0, 1, 3, 6, 9, 12])


# ── I/O helpers ──────────────────────────────────────────────────────────────

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


# ── Cargar datos ──────────────────────────────────────────────────────────────

Vm_am_reps = np.array([
    [vm_from_log(AM_LOGS / f"stage1_x{x}_r{r}_PMMCS_rc8p0.lammps",
                 AM_DATA / f"AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data",
                 tail=3)
     for r in (1, 2, 3)]
    for x in X_ALL
])
Vm_am_mean = Vm_am_reps.mean(axis=1)
Vm_am_std  = Vm_am_reps.std(axis=1, ddof=1)

Vm_iex1_reps = np.array([
    [vm_from_log(IEX1_LOGS / f"stage3_xt{x}_r{r}_PMMCS_rc8p0.lammps",
                 IEX1_DATA / f"IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data",
                 tail=200)
     for r in (1, 2, 3)]
    for x in X_IEX1
])
Vm_iex1_mean = Vm_iex1_reps.mean(axis=1)
Vm_iex1_std  = Vm_iex1_reps.std(axis=1, ddof=1)

Vm_iex2_reps = np.array([
    [vm_from_log(IEX2_LOGS / f"stage4_xp{x}_r{r}_PMMCS_rc8p0.lammps",
                 IEX2_DATA / f"IOX2_723K_xp{x}_r{r}_PMMCS_rc8p0.data",
                 tail=200)
     for r in (1, 2, 3)]
    for x in X_IEX2
])
Vm_iex2_mean = Vm_iex2_reps.mean(axis=1)
Vm_iex2_std  = Vm_iex2_reps.std(axis=1, ddof=1)

# ── Tabla ─────────────────────────────────────────────────────────────────────

print(f"\n{'x':>4}  {'Vm_AM (cm³/mol)':>18}  {'±':>7}  (tail=3)")
for x, m, s in zip(X_ALL, Vm_am_mean, Vm_am_std):
    print(f"  {x:>2d}   {m:>18.4f}  {s:>7.4f}")

print(f"\n{'x_t':>4}  {'Vm_IEX1 (cm³/mol)':>18}  {'±':>7}  (tail=200)")
for x, m, s in zip(X_IEX1, Vm_iex1_mean, Vm_iex1_std):
    print(f"  {x:>2d}   {m:>18.4f}  {s:>7.4f}")

print(f"\n{'x_p':>4}  {'Vm_IEX2 (cm³/mol)':>18}  {'±':>7}  (tail=200)")
for x, m, s in zip(X_IEX2, Vm_iex2_mean, Vm_iex2_std):
    print(f"  {x:>2d}   {m:>18.4f}  {s:>7.4f}")

# ── Figura ────────────────────────────────────────────────────────────────────

C_AM   = '#1a6bb0'
C_IEX1 = '#c0392b'
C_IEX2 = '#27ae60'
KW = dict(ms=7, lw=1.5, capsize=3, capthick=1.0, elinewidth=0.8)

fig, ax = plt.subplots(figsize=(6.5, 5.5))

ax.errorbar(X_ALL,  Vm_am_mean,   yerr=Vm_am_std,
            color=C_AM,  marker='o', label='As-melted', zorder=3, **KW)
ax.errorbar(X_IEX1, Vm_iex1_mean, yerr=Vm_iex1_std,
            color=C_IEX1, marker='s', label='IEX1', zorder=4, **KW)
ax.errorbar(X_IEX2, Vm_iex2_mean, yerr=Vm_iex2_std,
            color=C_IEX2, marker='D', label='IEX2', zorder=4, **KW)

ax.set_xlabel(r'$x$ / $x_\mathrm{target}$ / $x_\mathrm{parent}$ (mol% K$_2$O)', fontsize=10.5)
ax.set_ylabel(r'Molar Volume (cm$^3$/mol)', fontsize=11)
ax.set_xlim(-0.8, 16)
ax.set_xticks([0, 1, 3, 6, 9, 12, 15])
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.grid(axis='y', lw=0.4, color='#cccccc', zorder=0)
ax.grid(axis='y', which='minor', lw=0.2, color='#eeeeee', zorder=0)
ax.tick_params(labelsize=9.5)
ax.legend(fontsize=9.5, frameon=True, framealpha=0.9, edgecolor='#cccccc')

ax.set_title(
    r'PMMCS $r_c=8$ Å — Molar volume, 723 K'
    '\n'
    r'75SiO$_2\cdot$(15$-x$)Na$_2$O$\cdot x$K$_2$O$\cdot$10CaO',
    fontsize=10
)

fig.tight_layout()
fig.savefig(HERE / 'fig_molar_volume.pdf', dpi=300, bbox_inches='tight')
fig.savefig(HERE / 'fig_molar_volume.png', dpi=150, bbox_inches='tight')
print(f"\nGuardado: {HERE / 'fig_molar_volume.pdf'}")
plt.close(fig)
