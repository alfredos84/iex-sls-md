"""
fig_LNDC_IEX1.py
LNDC (Linear Network Dilation Coefficient) vs x — IEX Protocol 1
Sistema: 75SiO2·(15-x)Na2O·xK2O·10CaO (mol%), PMMCS rc=8 Å, 723 K

Fórmula (Vargheese 2014):
  LNDC_IEX  [ppk/mol%] = (ln(Vm_IEX(x_t)) - ln(Vm_AM(0))) / (3 * x_t) * 1000
  LNDC_AM   [ppk/mol%] = slope(ln Vm_AM vs x) / 3 * 1000   (ajuste lineal)

Ajuste logarítmico sobre los puntos IEX: LNDC ~ a*ln(x) + b
Eje secundario: % del LNDC as-melted

Nota: as-melted usa tail=3 (últimas 3 entradas thermo ≈ 718–738 K al final del
quench). IEX1 usa tail=200 (últimas 2000 ps de NPT a 723 K).
"""

import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

HERE     = Path(__file__).parent
BASE     = HERE.parent

AM_DATA  = BASE / "STAGE1_MELTQUENCH" / "data" / "asmelted_723K"
AM_LOGS  = BASE / "STAGE1_MELTQUENCH" / "logs"
IEX_DATA = BASE / "STAGE3_IEX_PROTO1" / "data" / "iox1_723K"
IEX_LOGS = BASE / "STAGE3_IEX_PROTO1" / "logs"

NA     = 6.02214076e23
N_MOLS = 5000

X_ALL  = np.array([0, 1, 3, 6, 9, 12, 15])
X_IEX1 = np.array([1, 3, 6, 9, 12, 15])


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

Vm_am = np.array([
    np.mean([
        vm_from_log(
            AM_LOGS  / f"stage1_x{x}_r{r}_PMMCS_rc8p0.lammps",
            AM_DATA  / f"AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data",
            tail=3,
        )
        for r in (1, 2, 3)
    ])
    for x in X_ALL
])

Vm_iex1 = np.array([
    np.mean([
        vm_from_log(
            IEX_LOGS / f"stage3_xt{x}_r{r}_PMMCS_rc8p0.lammps",
            IEX_DATA / f"IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data",
            tail=200,
        )
        for r in (1, 2, 3)
    ])
    for x in X_IEX1
])

# ── LNDC ─────────────────────────────────────────────────────────────────────

lnV_am      = np.log(Vm_am)
slope_am, _ = np.polyfit(X_ALL, lnV_am, 1)
LNDC_am     = slope_am / 3.0 * 1000          # ppk/mol%

LNDC_iex1 = np.array([
    (np.log(Vm_iex1[i]) - np.log(Vm_am[0])) / (3.0 * X_IEX1[i]) * 1000
    for i in range(len(X_IEX1))
])
pct_iex1 = LNDC_iex1 / LNDC_am * 100.0

print(f"\nLNDC As-Melted (ajuste lineal): {LNDC_am:.4f} ppk/mol%")
print(f"\n{'x_t':>4}  {'LNDC_IEX1':>12}  {'% AM':>8}")
for x, l, p in zip(X_IEX1, LNDC_iex1, pct_iex1):
    print(f"  {x:>2d}   {l:>12.4f}  {p:>8.2f}")

# ── Ajuste logarítmico sobre puntos IEX1 ─────────────────────────────────────

a, b   = np.polyfit(np.log(X_IEX1), LNDC_iex1, 1)
x_fit  = np.linspace(X_IEX1.min(), X_IEX1.max(), 300)
y_fit  = a * np.log(x_fit) + b
print(f"\nAjuste log: LNDC = {a:.4f}·ln(x) + {b:.4f}")

# ── Figura ────────────────────────────────────────────────────────────────────

C_AM   = '#1a6bb0'   # azul — as-melted
C_IEX1 = '#c0392b'  # rojo — IEX Proto1

ymax = max(LNDC_am, LNDC_iex1.max()) * 1.30

fig, ax = plt.subplots(figsize=(6.5, 5.5))
fig.subplots_adjust(left=0.15, right=0.82, top=0.92, bottom=0.13)

ax.axhline(LNDC_am, color=C_AM, linestyle='--', linewidth=2.0,
           label='As-melted', zorder=2)

ax.plot(X_IEX1, LNDC_iex1, marker='s', ms=7.5, lw=0,
        color=C_IEX1, label='IEX1', zorder=4)

ax.plot(x_fit, y_fit, color='#333333', lw=1.5, zorder=3,
        label='_nolegend_')

ax.set_xlabel(r'$x$ (mol%)', fontsize=11)
ax.set_ylabel(r'LNDC (ppk/mol%)', fontsize=11)
ax.set_xlim(-0.5, 16)
ax.set_xticks([0, 3, 6, 9, 12, 15])
ax.set_ylim(0, ymax)
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.grid(axis='y', lw=0.4, color='#cccccc', zorder=0)
ax.grid(axis='y', which='minor', lw=0.2, color='#eeeeee', zorder=0)
ax.tick_params(labelsize=9.5)
ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='#cccccc',
          loc='upper left')

# secondary axis: % of as-melted LNDC
ax2 = ax.twinx()
ax2.set_ylim(0, ymax / LNDC_am * 100.0)
pct_ticks = [0, 20, 40, 60, 80, 100]
ax2.set_yticks(pct_ticks)
ax2.set_yticklabels([f'{p}%' for p in pct_ticks], fontsize=9)
ax2.set_ylabel('% of As-melted LNDC', fontsize=10)
ax2.tick_params(labelsize=9)

out_pdf = HERE / "fig_LNDC_IEX1.pdf"
out_png = HERE / "fig_LNDC_IEX1.png"
fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"\nGuardado: {out_pdf}")
print(f"Guardado: {out_png}")
plt.close(fig)
