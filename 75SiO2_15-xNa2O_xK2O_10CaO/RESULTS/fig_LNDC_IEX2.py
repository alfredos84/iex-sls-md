"""
fig_LNDC_IEX2.py
LNDC (Linear Network Dilation Coefficient) vs C_exc — IEX Protocol 2
Sistema: 75SiO2·(15-x)Na2O·xK2O·10CaO (mol%), PMMCS rc=8 Å, 723 K

IEX Proto2: desde cada vidrio as-melted x_parent se reemplazan TODOS los Na→K.
La concentración intercambiada es C_exc = 15 - x_parent (mol%).

Fórmula:
  LNDC_IEX2  [ppk/mol%] = (ln(Vm_IEX2(xp)) - ln(Vm_AM(xp))) / (3 * C_exc) * 1000
  LNDC_AM    [ppk/mol%] = slope(ln Vm_AM vs x) / 3 * 1000   (ajuste lineal)

Ajuste logarítmico sobre los puntos IEX2: LNDC ~ a*ln(C_exc) + b
Eje secundario: % del LNDC as-melted
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
IEX2_DATA = BASE / "STAGE4_IEX_PROTO2" / "data" / "iox2_723K"
IEX2_LOGS = BASE / "STAGE4_IEX_PROTO2" / "logs"

NA     = 6.02214076e23
N_MOLS = 5000

X_ALL  = np.array([0, 1, 3, 6, 9, 12, 15])
X_IEX2 = np.array([0, 1, 3, 6, 9, 12])      # x_parent


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
            AM_LOGS / f"stage1_x{x}_r{r}_PMMCS_rc8p0.lammps",
            AM_DATA / f"AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data",
            tail=3,
        )
        for r in (1, 2, 3)
    ])
    for x in X_ALL
])

Vm_iex2 = np.array([
    np.mean([
        vm_from_log(
            IEX2_LOGS / f"stage4_xp{x}_r{r}_PMMCS_rc8p0.lammps",
            IEX2_DATA / f"IOX2_723K_xp{x}_r{r}_PMMCS_rc8p0.data",
            tail=200,
        )
        for r in (1, 2, 3)
    ])
    for x in X_IEX2
])

# Vm_AM en cada x_parent
am_at_xp = {x: Vm_am[i] for i, x in enumerate(X_ALL) if x in X_IEX2}

# ── LNDC ─────────────────────────────────────────────────────────────────────

lnV_am      = np.log(Vm_am)
slope_am, _ = np.polyfit(X_ALL, lnV_am, 1)
LNDC_am     = slope_am / 3.0 * 1000          # ppk/mol%

# C_exc = 15 - x_parent (mol% intercambiado)
C_exc = 15 - X_IEX2   # [15, 14, 12, 9, 6, 3]

LNDC_iex2 = np.array([
    (np.log(Vm_iex2[i]) - np.log(am_at_xp[X_IEX2[i]])) / (3.0 * C_exc[i]) * 1000
    for i in range(len(X_IEX2))
])
pct_iex2 = LNDC_iex2 / LNDC_am * 100.0

print(f"\nLNDC As-Melted (ajuste lineal): {LNDC_am:.4f} ppk/mol%")
print(f"\n{'xp':>4}  {'C_exc':>6}  {'LNDC_IEX2':>12}  {'% AM':>8}")
for xp, c, l, p in zip(X_IEX2, C_exc, LNDC_iex2, pct_iex2):
    print(f"  {xp:>2d}   {c:>6.0f}  {l:>12.4f}  {p:>8.2f}")

# ── Ajuste logarítmico sobre C_exc ───────────────────────────────────────────

a, b   = np.polyfit(np.log(C_exc), LNDC_iex2, 1)
c_fit  = np.linspace(C_exc.min(), C_exc.max(), 300)
y_fit  = a * np.log(c_fit) + b
print(f"\nAjuste log: LNDC = {a:.4f}·ln(C_exc) + {b:.4f}")

# ── Figura ────────────────────────────────────────────────────────────────────

C_AM   = '#1a6bb0'   # azul  — as-melted
C_IEX2 = '#27ae60'  # verde — IEX2

ymax = max(LNDC_am, LNDC_iex2.max()) * 1.30

fig, ax = plt.subplots(figsize=(6.5, 5.5))
fig.subplots_adjust(left=0.15, right=0.82, top=0.92, bottom=0.13)

ax.axhline(LNDC_am, color=C_AM, linestyle='--', linewidth=2.0,
           label='As-melted', zorder=2)

ax.plot(C_exc, LNDC_iex2, marker='s', ms=7.5, lw=0,
        color=C_IEX2, label='IEX2', zorder=4)

ax.plot(c_fit, y_fit, color='#333333', lw=1.5, zorder=3,
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


out_pdf = HERE / "fig_LNDC_IEX2.pdf"
out_png = HERE / "fig_LNDC_IEX2.png"
fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"\nGuardado: {out_pdf}")
print(f"Guardado: {out_png}")
plt.close(fig)
