"""
fig_arrhenius_MSD.py
Difusividades Na y K + coeficiente de interdifusión Darken → Arrhenius
Sistema: 75SiO2·(15-x)Na2O·xK2O·10CaO, PMMCS rc=8 Å

STAGE10: T = 600, 723, 800, 1000, 1200 K | eq=1ns | prod=4ns
MSD guardado cada 1000 steps (dt=0.002 ps) → cada 2 ps

Análisis:
  1. Ajuste lineal MSD_tot vs t (último 75% de la producción) → D = slope/6
  2. D̃ (Darken) = f_Na·D_K + f_K·D_Na  (f_i = fracción molar alcalina)
  3. Arrhenius: ln(D) vs 1/T → Ea = −slope·k_B

Unidades:
  MSD en Å², t en ps → D en Å²/ps → ×1e-8 para m²/s
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.stats import linregress

HERE     = Path(__file__).parent
MSD_DIR  = HERE.parent / "STAGE10_MSD" / "data"

# T_LIST   = [600, 723, 800, 1000, 1200]   # K
T_LIST   = [723, 800, 1000, 1200]   # K
X_LIST   = [1, 3, 6, 9, 12]
REPS     = [1, 2, 3]
DT_PS    = 0.002                          # ps por step
KB_EV    = 8.617333e-5                    # eV/K

# Fracción molar de Na y K entre los iones alcalinos
def alkali_fractions(x):
    f_K  = x / 15.0
    f_Na = (15 - x) / 15.0
    return f_Na, f_K


# ── Leer MSD y ajustar ────────────────────────────────────────────────────────

def load_msd(species, T, x, r):
    path = MSD_DIR / f"msd_{species}_{T}K_x{x}_r{r}.dat"
    if not path.exists():
        return None, None
    data = np.loadtxt(path, comments='#')
    steps = data[:, 0]
    msd   = data[:, 4]      # columna msd_tot
    t_ps  = steps * DT_PS
    return t_ps, msd


def diffusivity(t_ps, msd, fit_frac=0.75):
    """Ajuste lineal sobre el último fit_frac de la producción. D en Å²/ps."""
    if t_ps is None:
        return np.nan
    # excluir step=0 (msd≈0) y usar solo la fracción final
    mask = t_ps > t_ps[-1] * (1 - fit_frac)
    if mask.sum() < 10:
        return np.nan
    slope, _, _, _, _ = linregress(t_ps[mask], msd[mask])
    return slope / 6.0   # Å²/ps


def D_m2s(D_angps):
    return D_angps * 1e-8   # 1 Å²/ps = 1e-8 m²/s


# ── Calcular D para todas las combinaciones ───────────────────────────────────

# D[species][x][T] = (mean, std) en m²/s
results = {sp: {x: {} for x in X_LIST} for sp in ('Na', 'K')}

print(f"\n{'T':>5} {'x':>3} {'D_Na (m²/s)':>14} {'D_K (m²/s)':>14} {'D~_NK (m²/s)':>15}")
print("-" * 58)

D_interp = {x: {} for x in X_LIST}   # interdifusión Darken

for x in X_LIST:
    f_Na, f_K = alkali_fractions(x)
    for T in T_LIST:
        D_na_reps, D_k_reps = [], []
        for r in REPS:
            t, msd = load_msd('Na', T, x, r)
            D_na_reps.append(D_m2s(diffusivity(t, msd)))
            t, msd = load_msd('K', T, x, r)
            D_k_reps.append(D_m2s(diffusivity(t, msd)))

        D_na_reps = np.array(D_na_reps)
        D_k_reps  = np.array(D_k_reps)

        valid_na = D_na_reps[np.isfinite(D_na_reps)]
        valid_k  = D_k_reps[np.isfinite(D_k_reps)]

        dna_mean = valid_na.mean() if len(valid_na) else np.nan
        dna_std  = valid_na.std()  if len(valid_na) > 1 else np.nan
        dk_mean  = valid_k.mean()  if len(valid_k)  else np.nan
        dk_std   = valid_k.std()   if len(valid_k)  > 1 else np.nan

        results['Na'][x][T] = (dna_mean, dna_std)
        results['K'][x][T]  = (dk_mean,  dk_std)

        # Nernst-Planck: D̃ = (D_Na·D_K) / (D_Na·f_Na + D_K·f_K)
        dint = (dna_mean * dk_mean) / (dna_mean * f_Na + dk_mean * f_K)
        D_interp[x][T] = dint

        print(f"{T:>5} {x:>3} {dna_mean:>14.3e} {dk_mean:>14.3e} {dint:>15.3e}")


# ── Arrhenius fits ────────────────────────────────────────────────────────────

def arrhenius_fit(T_list, D_list):
    """ln(D) vs 1/T. Retorna Ea (eV), D0 (m²/s), inv_T_fit, lnD_fit."""
    inv_T = np.array([1.0 / T for T in T_list])
    lnD   = np.array([np.log(d) for d in D_list])
    mask  = np.isfinite(lnD)
    if mask.sum() < 2:
        return np.nan, np.nan, None, None
    slope, intercept, _, _, _ = linregress(inv_T[mask], lnD[mask])
    Ea   = -slope * KB_EV
    D0   = np.exp(intercept)
    inv_T_fit = np.linspace(inv_T[mask].min(), inv_T[mask].max(), 100)
    lnD_fit   = slope * inv_T_fit + intercept
    return Ea, D0, inv_T_fit, lnD_fit


print(f"\n{'x':>3} {'Ea_Na (eV)':>12} {'Ea_K (eV)':>11} {'Ea_NK (eV)':>12}")
print("-" * 42)

Ea_Na_all, Ea_K_all, Ea_NK_all = {}, {}, {}
fits = {'Na': {}, 'K': {}, 'NK': {}}

for x in X_LIST:
    D_na_list = [results['Na'][x][T][0] for T in T_LIST]
    D_k_list  = [results['K'][x][T][0]  for T in T_LIST]
    D_nk_list = [D_interp[x][T]         for T in T_LIST]

    Ea_na, _, it_na, ld_na = arrhenius_fit(T_LIST, D_na_list)
    Ea_k,  _, it_k,  ld_k  = arrhenius_fit(T_LIST, D_k_list)
    Ea_nk, _, it_nk, ld_nk = arrhenius_fit(T_LIST, D_nk_list)

    Ea_Na_all[x] = Ea_na
    Ea_K_all[x]  = Ea_k
    Ea_NK_all[x] = Ea_nk

    fits['Na'][x] = (it_na, ld_na)
    fits['K'][x]  = (it_k,  ld_k)
    fits['NK'][x] = (it_nk, ld_nk)

    print(f"{x:>3} {Ea_na:>12.3f} {Ea_k:>11.3f} {Ea_nk:>12.3f}")


# ── Colores por composición ────────────────────────────────────────────────────

COLORS = {1: '#1a6bb0', 3: '#27ae60', 6: '#e67e22', 9: '#8e44ad', 12: '#c0392b'}
MARKERS = {1: 'o', 3: 's', 6: '^', 9: 'D', 12: 'v'}

inv_T_ticks = [1000/T for T in T_LIST]


# ── Fig 1 — Arrhenius D_Na, D_K, D̃  (con barras de error) ──────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=False)
fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.13, wspace=0.32)

titles    = [r'$D_\mathrm{Na}$', r'$D_\mathrm{K}$', r'$\tilde{D}_\mathrm{Na-K}$ (Nernst-Planck)']
data_keys = ['Na', 'K', 'NK']

INV_T_MAX_PLOT = 1000.0 / min(T_LIST) * 1.08   # margen del 8% más allá del punto más frío
INV_T_EXT     = INV_T_MAX_PLOT

for ax, key, title in zip(axes, data_keys, titles):
    for x in X_LIST:
        c, m = COLORS[x], MARKERS[x]
        if key == 'NK':
            D_list = [D_interp[x][T] for T in T_LIST]
            yerr_lo = yerr_hi = None
        else:
            D_list  = [results[key][x][T][0] for T in T_LIST]
            std_list = [results[key][x][T][1] for T in T_LIST]

        inv_T  = np.array([1000.0/T for T in T_LIST])
        D_arr  = np.array(D_list)
        mask   = np.isfinite(D_arr) & (D_arr > 0)
        if not mask.any():
            continue

        ivT = inv_T[mask]
        dv  = D_arr[mask]

        if key != 'NK':
            std_arr = np.array(std_list)
            std_v   = np.where(np.isfinite(std_arr[mask]), std_arr[mask], 0.0)
            lo_err  = np.clip(std_v, 0, dv * 0.99)   # no bajar de 0
            hi_err  = std_v
            ax.errorbar(ivT, dv, yerr=[lo_err, hi_err],
                        fmt=m, ms=6.5, lw=0, color=c, label=f'$x={x}$',
                        elinewidth=1.2, capsize=3, capthick=1.1, zorder=4)
        else:
            ax.semilogy(ivT, dv, marker=m, ms=6.5, lw=0, color=c,
                        label=f'$x={x}$', zorder=4)

        # Recta Arrhenius extendida desde 0 hasta INV_T_EXT (en unidades de 1/T)
        it_fit_raw, ld_fit = fits[key][x]
        if it_fit_raw is not None:
            it_ext = np.linspace(0.0, INV_T_EXT / 1000.0, 300)
            slope_fit = (ld_fit[-1] - ld_fit[0]) / (it_fit_raw[-1] - it_fit_raw[0])
            intercept_fit = ld_fit[0] - slope_fit * it_fit_raw[0]
            ld_ext = slope_fit * it_ext + intercept_fit
            ax.semilogy(it_ext * 1000, np.exp(ld_ext), color=c, lw=1.3,
                        alpha=0.65, zorder=3)

    ax.set_xlabel(r'$1000\,/\,T$ (K$^{-1}$)', fontsize=11)
    ax.set_ylabel(r'$D$ (m$^2$/s)', fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0, INV_T_MAX_PLOT)
    # Ticks en los datos + el origen
    data_ticks = [1000/T for T in T_LIST]
    ax.set_xticks([0] + data_ticks)
    ax.set_xticklabels(['0'] + [f'{t:.2f}' for t in data_ticks], fontsize=8.5)
    ax.tick_params(labelsize=9)
    ax.grid(axis='y', lw=0.35, color='#dddddd', zorder=0)
    ax.grid(axis='x', lw=0.25, color='#eeeeee', zorder=0)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='#cccccc')

fig.suptitle(
    r'PMMCS $r_c=8$ Å — Arrhenius: self-diffusion and interdiffusion  |  723 K glass'
    '\n'
    r'75SiO$_2\cdot$(15$-x$)Na$_2$O$\cdot x$K$_2$O$\cdot$10CaO',
    fontsize=10
)

out1 = HERE / "fig_arrhenius_MSD.pdf"
fig.savefig(out1, dpi=300, bbox_inches='tight')
fig.savefig(HERE / "fig_arrhenius_MSD.png", dpi=150, bbox_inches='tight')
print(f"\nGuardado: {out1}")
plt.close(fig)


# ── Fig 2 — Ea_Na, Ea_K, Ea_NK vs x (doble eje eV / kJ/mol) ─────────────────

EV_TO_KJ = 96.485   # 1 eV = 96.485 kJ/mol

x_arr  = np.array(X_LIST)
ea_na  = np.array([Ea_Na_all[x] for x in X_LIST])
ea_k   = np.array([Ea_K_all[x]  for x in X_LIST])
ea_nk  = np.array([Ea_NK_all[x] for x in X_LIST])

fig2, ax2 = plt.subplots(figsize=(6.5, 5.5))
fig2.subplots_adjust(left=0.13, right=0.87, top=0.95, bottom=0.13)

ax2.plot(x_arr, ea_na, marker='o', ms=7,   lw=1.8, color='#1a6bb0',
         label=r'$D_\mathrm{Na}$', zorder=4)
ax2.plot(x_arr, ea_k,  marker='s', ms=7,   lw=1.8, color='#c0392b',
         label=r'$D_\mathrm{K}$',  zorder=4)
ax2.plot(x_arr, ea_nk, marker='^', ms=7.5, lw=1.8, color='#27ae60',
         label=r'$\tilde{D}_\mathrm{Na-K}$ (N-P)', zorder=4)

ax2.set_xlabel(r'$x$ (mol% K$_2$O)', fontsize=11)
ax2.set_ylabel(r'$E_a$ (eV)', fontsize=11)
ax2.set_xticks(X_LIST)
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax2.grid(axis='y', lw=0.35, color='#dddddd', zorder=0)
ax2.tick_params(labelsize=9.5)
ax2.legend(fontsize=9.5, frameon=True, framealpha=0.9, edgecolor='#cccccc')

# Eje derecho en kJ/mol — sincronizado manualmente después de dibujar
fig2.canvas.draw()
ymin, ymax = ax2.get_ylim()
ax2r = ax2.twinx()
ax2r.set_ylim(ymin * EV_TO_KJ, ymax * EV_TO_KJ)
ax2r.set_ylabel(r'$E_a$ (kJ/mol)', fontsize=11)
ax2r.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax2r.tick_params(labelsize=9.5)

out2 = HERE / "fig_Ea_vs_x.pdf"
fig2.savefig(out2, dpi=300, bbox_inches='tight')
fig2.savefig(HERE / "fig_Ea_vs_x.png", dpi=150, bbox_inches='tight')
print(f"Guardado: {out2}")
plt.close(fig2)
