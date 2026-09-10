"""
fig_stress.py
Distribuciones de estrés hidrostático atómico P_i = -(sxx+syy+szz)/(3*V_i) [GPa]
separadas en dos grupos:
  — Modificadores : Na, K, Ca   (PLIM_MOD)
  — Red           : Si, O       (PLIM_NET — escalas muy distintas)

Genera:
  fig_stress_XX{chi}.pdf  — 2 filas × 2 cols: (mod/red) × (CaO/Ca-free)
  fig_stress_mean.pdf     — <P> vs χ, mismo layout

Atom types: Si=1, O=2, Ca=3, Na=4, K=5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.stats import gaussian_kde

HERE     = Path(__file__).parent
DUMP_DIR = HERE / "dumps"
VORO_DIR = HERE.parent / "VORONOI"

TYPE_SI, TYPE_O, TYPE_CA, TYPE_NA, TYPE_K = 1, 2, 3, 4, 5

XX_CONFIG = {
    20:  (3,  3,  3,  5,  5,  5),
    40:  (6,  6,  6,  10, 10, 10),
    60:  (9,  9,  9,  15, 15, 15),
    80:  (12, 12, 12, 20, 20, 20),
    100: (0,  15, 15, 0,  25, 25),
}

CHI = np.array(sorted(XX_CONFIG.keys()), dtype=float)

PLIM_MOD = (-20,  20)    # GPa — Na, K, Ca
PLIM_NET = (-210, 90)    # GPa — Si (-120) y O (+24) en el mismo eje


# ── Lectura ───────────────────────────────────────────────────────────────────

def read_stress(path):
    data = {}
    in_atoms = False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("ITEM: ATOMS"): in_atoms = True;  continue
            if s.startswith("ITEM:"):       in_atoms = False; continue
            if in_atoms and s:
                p = s.split()
                data[int(p[0])] = (int(p[1]), float(p[2]), float(p[3]), float(p[4]))
    return data


def read_vol(path):
    data = {}
    in_atoms = False
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s.startswith("ITEM: ATOMS"): in_atoms = True;  continue
            if s.startswith("ITEM:"):       in_atoms = False; continue
            if in_atoms and s:
                p = s.split()
                data[int(p[0])] = (int(p[1]), float(p[2]))
    return data


def pressure_per_type(stress_path, vol_path):
    stress = read_stress(stress_path)
    vol    = read_vol(vol_path)
    by_type = {t: [] for t in (TYPE_SI, TYPE_O, TYPE_CA, TYPE_NA, TYPE_K)}
    for aid, (t, sxx, syy, szz) in stress.items():
        if aid not in vol or t not in by_type:
            continue
        _, v = vol[aid]
        if v <= 0:
            continue
        by_type[t].append(-(sxx + syy + szz) / (3.0 * v) * 1e-4)   # GPa
    return {t: np.array(vals) for t, vals in by_type.items()}


def pool_pressure(tags):
    combined = {t: [] for t in (TYPE_SI, TYPE_O, TYPE_CA, TYPE_NA, TYPE_K)}
    for tag in tags:
        sp = DUMP_DIR / f"{tag}_stress.dat"
        vp = VORO_DIR / f"{tag}_vol.dat"
        if not sp.exists() or not vp.exists():
            print(f"  AVISO: falta {tag}")
            continue
        for t, arr in pressure_per_type(sp, vp).items():
            combined[t].extend(arr)
    return {t: np.array(v) for t, v in combined.items()}


# ── Plot helpers ──────────────────────────────────────────────────────────────

plt.rcParams.update({
    'font.family':     'Times New Roman',
    'font.size':       17,
    'axes.linewidth':  0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

# Modificadores
C_NA = '#1a3a5c';  LS_NA = '-'
C_K  = '#2ca0c4';  LS_K  = '--'
C_CA = '#7b3f00';  LS_CA = '-.'
# Red
C_SI = '#8e44ad';  LS_SI = '-'
C_O  = '#27ae60';  LS_O  = '--'

MOD_SERIES = [
    (TYPE_NA, C_NA, LS_NA, 'Na'),
    (TYPE_K,  C_K,  LS_K,  'K'),
    (TYPE_CA, C_CA, LS_CA, 'Ca'),
]
NET_SERIES = [
    (TYPE_SI, C_SI, LS_SI, 'Si'),
    (TYPE_O,  C_O,  LS_O,  'O'),
]


def kde_plot(ax, arr, color, ls, label, x_grid):
    if len(arr) < 5:
        return
    kde = gaussian_kde(arr, bw_method='silverman')
    y   = kde(x_grid)
    ymax = y.max()
    if ymax == 0:
        return
    ax.plot(x_grid, y / ymax, color=color, ls=ls, lw=1.8, label=label)


def decorate(ax, xlabel, xlim):
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('Normalized KDE', fontsize=16)
    ax.axvline(0, color='#aaaaaa', lw=0.8, ls=':', zorder=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.tick_params(labelsize=14, which='both')
    ax.grid(lw=0.35, color='#dddddd', zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8)


# ── Recopilar datos ───────────────────────────────────────────────────────────

results = {
    'cao':  {'am': {}, 'iex': {}},
    'noca': {'am': {}, 'iex': {}},
}

for XX, (xna_cao, xka_cao, xt_cao, xna_noca, xka_noca, xt_noca) in sorted(XX_CONFIG.items()):
    print(f"XX={XX}%...")
    results['cao']['am'][XX]   = pool_pressure([f"cao_am_x{xka_cao}_r{r}"    for r in (1,2,3)])
    results['cao']['iex'][XX]  = pool_pressure([f"cao_iex1_xt{xt_cao}_r{r}"  for r in (1,2,3)])
    results['noca']['am'][XX]  = pool_pressure([f"noca_am_x{xka_noca}_r{r}"  for r in (1,2,3)])
    results['noca']['iex'][XX] = pool_pressure([f"noca_iex1_xt{xt_noca}_r{r}" for r in (1,2,3)])


# ── Figura 1: distribuciones por χ (2×2) ─────────────────────────────────────

x_mod = np.linspace(*PLIM_MOD, 1000)
x_net = np.linspace(*PLIM_NET, 2000)

for XX in sorted(XX_CONFIG.keys()):
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))

    for col, (sys, title_base) in enumerate([('cao', '10CaO'), ('noca', 'Ca-free')]):
        am  = results[sys]['am'][XX]
        iex = results[sys]['iex'][XX]

        # Fila 0 — Modificadores
        ax0 = axes[0, col]
        for t, color, ls, name in MOD_SERIES:
            if sys == 'noca' and t == TYPE_CA:
                continue
            kde_plot(ax0, am[t],  color, '-',  f'{name} (AM)',  x_mod)
            kde_plot(ax0, iex[t], color, '--', f'{name} (IEX)', x_mod)
        ax0.set_title(rf'{title_base} — $\chi$ = {XX}%', fontsize=16)
        decorate(ax0, 'Hydrostatic Stress (GPa)', PLIM_MOD)
        ax0.legend(fontsize=12, frameon=True, framealpha=0.9,
                   edgecolor='#cccccc', loc='upper right', ncol=2)

        # Fila 1 — Red
        ax1 = axes[1, col]
        for t, color, ls, name in NET_SERIES:
            kde_plot(ax1, am[t],  color, '-',  f'{name} (AM)',  x_net)
            kde_plot(ax1, iex[t], color, '--', f'{name} (IEX)', x_net)
        decorate(ax1, 'Hydrostatic Stress (GPa)', PLIM_NET)
        ax1.legend(fontsize=12, frameon=True, framealpha=0.9,
                   edgecolor='#cccccc', loc='upper right', ncol=2)

    # Etiquetas de fila
    axes[0, 0].set_ylabel('Modifiers — Normalized KDE', fontsize=16)
    axes[1, 0].set_ylabel('Network — Normalized KDE',   fontsize=16)
    axes[0, 1].set_ylabel('')
    axes[1, 1].set_ylabel('')

    fig.tight_layout()
    out = HERE / f'fig_stress_XX{XX}.pdf'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=150, bbox_inches='tight')
    print(f"  Guardado: {out.name}")
    plt.close(fig)


# ── Figura 2: <P> vs χ (2×2) ─────────────────────────────────────────────────

fig2, axes2 = plt.subplots(2, 2, figsize=(13, 13))

for col, (sys, title) in enumerate([('cao', r'10CaO'), ('noca', r'Ca-free')]):

    # Fila 0 — Modificadores
    ax0 = axes2[0, col]
    for t, color, ls, name in MOD_SERIES:
        if sys == 'noca' and t == TYPE_CA:
            continue
        mean_am  = np.array([results[sys]['am'][XX][t].mean()
                              if len(results[sys]['am'][XX][t]) > 0 else np.nan
                              for XX in sorted(XX_CONFIG)])
        mean_iex = np.array([results[sys]['iex'][XX][t].mean()
                              if len(results[sys]['iex'][XX][t]) > 0 else np.nan
                              for XX in sorted(XX_CONFIG)])
        mask_am  = ~np.isnan(mean_am)
        mask_iex = ~np.isnan(mean_iex)
        ax0.plot(CHI[mask_am],  mean_am[mask_am],  color=color, marker='o',
                 ls='-',  ms=6, lw=1.4, label=f'{name} (AM)')
        ax0.plot(CHI[mask_iex], mean_iex[mask_iex], color=color, marker='s',
                 ls='--', ms=6, lw=1.4, label=f'{name} (IEX)')

    ax0.axhline(0, color='#aaaaaa', lw=0.8, ls=':', zorder=0)
    ax0.set_title(title, fontsize=16)
    ax0.set_xlabel(r'$\chi$ (%)', fontsize=16)
    ax0.set_ylabel(r'$\langle P \rangle$ (GPa)', fontsize=16)
    ax0.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax0.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax0.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax0.tick_params(labelsize=14, which='both')
    ax0.grid(lw=0.35, color='#dddddd', zorder=0)
    ax0.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    for sp in ax0.spines.values(): sp.set_visible(True); sp.set_linewidth(0.8)

    # Fila 1 — Red
    ax1 = axes2[1, col]
    for t, color, ls, name in NET_SERIES:
        mean_am  = np.array([results[sys]['am'][XX][t].mean()
                              if len(results[sys]['am'][XX][t]) > 0 else np.nan
                              for XX in sorted(XX_CONFIG)])
        mean_iex = np.array([results[sys]['iex'][XX][t].mean()
                              if len(results[sys]['iex'][XX][t]) > 0 else np.nan
                              for XX in sorted(XX_CONFIG)])
        mask_am  = ~np.isnan(mean_am)
        mask_iex = ~np.isnan(mean_iex)
        ax1.plot(CHI[mask_am],  mean_am[mask_am],  color=color, marker='o',
                 ls='-',  ms=6, lw=1.4, label=f'{name} (AM)')
        ax1.plot(CHI[mask_iex], mean_iex[mask_iex], color=color, marker='s',
                 ls='--', ms=6, lw=1.4, label=f'{name} (IEX)')

    ax1.axhline(0, color='#aaaaaa', lw=0.8, ls=':', zorder=0)
    ax1.set_xlabel(r'$\chi$ (%)', fontsize=16)
    ax1.set_ylabel(r'$\langle P \rangle$ (GPa)', fontsize=16)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax1.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax1.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax1.tick_params(labelsize=14, which='both')
    ax1.grid(lw=0.35, color='#dddddd', zorder=0)
    ax1.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    for sp in ax1.spines.values(): sp.set_visible(True); sp.set_linewidth(0.8)

fig2.tight_layout()
out2 = HERE / 'fig_stress_mean.pdf'
fig2.savefig(out2, dpi=300, bbox_inches='tight')
fig2.savefig(out2.with_suffix('.png'), dpi=150, bbox_inches='tight')
print(f"\nGuardado: {out2.name}")
plt.close(fig2)

print("\nListo.")
