"""
fig_number_density_vargheese.py
Reproducción de Fig. 5 de Vargheese et al., J. Non-Cryst. Solids 403 (2014) 107.

Distribución de densidad numérica de TODOS los átomos dentro del radio del primer
mínimo de la RDF K-O (3.8 Å) alrededor de cada ion alcalino.

Tres series:
  1. Na en AM x=0  (extremo puro Na: 75SiO2·15Na2O·10CaO)
  2. K  en AM x=15 (extremo puro K:  75SiO2·15K2O·10CaO)
  3. K  en IEX1 xt=15 (intercambio TOTAL desde x=0 → mismo xf=15 que AM x=15)

Analogía con Vargheese2014:
  Na(AM) = Na en as-melted 20Na2O·20Al2O3·60SiO2
  K(AM)  = K en as-melted 20K2O·20Al2O3·60SiO2
  K(IEX) = K en IEX (completamente intercambiado desde Na-puro)

Resultado esperado: Na(AM) más denso > K(IEX) intermedio > K(AM) más abierto.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

HERE     = Path(__file__).parent
BASE     = HERE.parent
AM_DIR   = BASE / "STAGE1_MELTQUENCH" / "data" / "asmelted_723K"
IEX1_DIR = BASE / "STAGE3_IEX_PROTO1" / "data" / "iox1_723K"

R_CUT = 3.8   # Å — primer mínimo de la RDF K-O (Vargheese 2014)
V_SPHERE = (4 / 3) * np.pi * R_CUT ** 3

TYPE_Na, TYPE_K = 4, 5


# ── Parser ────────────────────────────────────────────────────────────────────

def read_data(path):
    box = {}
    atoms = []
    in_atoms = False
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            s = line.strip()
            if 'xlo xhi' in s:
                p = s.split(); box['x'] = float(p[1]) - float(p[0])
            elif 'ylo yhi' in s:
                p = s.split(); box['y'] = float(p[1]) - float(p[0])
            elif 'zlo zhi' in s:
                p = s.split(); box['z'] = float(p[1]) - float(p[0])
            elif s.startswith('Atoms'):
                in_atoms = True
            elif in_atoms and s:
                p = s.split()
                if len(p) >= 6:
                    try:
                        atoms.append((int(p[1]),
                                      float(p[3]), float(p[4]), float(p[5])))
                    except ValueError:
                        pass
    L = np.array([box['x'], box['y'], box['z']])
    atoms = np.array(atoms)
    pos = {}
    for t in np.unique(atoms[:, 0]).astype(int):
        pos[t] = atoms[atoms[:, 0] == t, 1:4]
    return L, pos


# ── Densidad numérica ─────────────────────────────────────────────────────────

def number_density(pos_center, pos_all, L, r_cut, batch=150):
    """
    Para cada átomo central, cuenta todos los átomos en pos_all a < r_cut
    (excluyendo el átomo central mismo si coincide con pos_all).
    Devuelve densidad = N_vecinos / V_esfera  en Å⁻³.
    """
    r2 = r_cut ** 2
    V  = (4 / 3) * np.pi * r_cut ** 3
    rho = []
    for i in range(0, len(pos_center), batch):
        pc = pos_center[i:i + batch]
        dr = pc[:, None, :] - pos_all[None, :, :]   # (B, N_all, 3)
        dr -= np.round(dr / L) * L
        d2 = (dr ** 2).sum(axis=2)                   # (B, N_all)
        n = ((d2 < r2) & (d2 > 1e-12)).sum(axis=1)   # exclude self
        rho.extend((n / V).tolist())
    return np.array(rho)


def collect_densities(path_fn, x_val, center_type, reps=(1, 2, 3)):
    """Acumula densidades numéricas de todas las réplicas."""
    all_rho = []
    for r in reps:
        p = path_fn(x_val, r)
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        L, pos = read_data(p)
        if center_type not in pos:
            continue
        pos_all = np.vstack(list(pos.values()))   # todos los átomos
        rho = number_density(pos[center_type], pos_all, L, R_CUT)
        all_rho.extend(rho.tolist())
        print(f"  {p.name}: N_center={len(pos[center_type])}, "
              f"mean_rho={rho.mean():.4f} Å⁻³")
    return np.array(all_rho)


def am_path(x, r):
    return AM_DIR / f"AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data"


def iex1_path(x, r):
    return IEX1_DIR / f"IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data"


# ── Calcular ──────────────────────────────────────────────────────────────────

print("Na en AM x=0 (puro Na)...")
rho_Na_AM = collect_densities(am_path, 0, TYPE_Na)

print("K en AM x=15 (puro K)...")
rho_K_AM  = collect_densities(am_path, 15, TYPE_K)

print("K en IEX1 xt=15 (intercambio total desde x=0)...")
rho_K_IEX = collect_densities(iex1_path, 15, TYPE_K)

print(f"\nResumen (densidad numérica en Å⁻³, r_cut={R_CUT} Å):")
for label, arr in [('Na AM x=0 ', rho_Na_AM),
                   ('K  AM x=15', rho_K_AM),
                   ('K  IEX xt=15', rho_K_IEX)]:
    if len(arr) > 0:
        print(f"  {label}: mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
              f"std={arr.std():.4f}, N={len(arr)}")


# ── Figura ────────────────────────────────────────────────────────────────────

C_Na_AM = '#1a6bb0'   # azul  — Na(AM)
C_K_AM  = '#c0392b'   # rojo  — K(AM)
C_K_IEX = '#27ae60'   # verde — K(IEX)

rho_max = max(rho_Na_AM.max(), rho_K_AM.max(), rho_K_IEX.max())

series = [
    (rho_Na_AM, C_Na_AM, 'o', r'Na (AM, $x=0$)'),
    (rho_K_AM,  C_K_AM,  's', r'K (AM, $x=15$)'),
    (rho_K_IEX, C_K_IEX, '^', r'K (IEX1, $x_t=15$)'),
]

fig, ax = plt.subplots(figsize=(6.5, 5.5))
fig.subplots_adjust(left=0.13, right=0.97, top=0.88, bottom=0.12)

for arr, color, marker, label in series:
    if len(arr) == 0:
        continue
    # distribución discreta exacta: fracción de iones con cada N entero
    N_arr   = np.round(arr * V_SPHERE).astype(int)
    counts  = np.bincount(N_arr)
    n_vals  = np.arange(len(counts))
    x_vals  = n_vals / V_SPHERE      # Å⁻³, valores exactos
    frac    = counts / counts.max()

    mask = counts > 0
    xv, yv = x_vals[mask], frac[mask]

    ax.fill_between(xv, yv, alpha=0.18, color=color, zorder=2)
    ax.plot(xv, yv, color=color, lw=1.5, zorder=3, label=label)
    ax.scatter(xv, yv, color=color, marker=marker, s=30, zorder=4,
               edgecolors='white', linewidths=0.5)

ax.set_xlabel(r'Number Density (Å$^{-3}$)', fontsize=11)
ax.set_ylabel('Normalized Amplitude', fontsize=11)
ax.set_xlim(0, rho_max * 1.08)
ax.set_ylim(0, 1.12)
ax.tick_params(labelsize=9.5)
ax.legend(fontsize=9.5, frameon=True, framealpha=0.9, edgecolor='#cccccc',
          loc='upper left')
ax.grid(axis='y', lw=0.35, color='#dddddd', zorder=0)

ax.set_title(
    r'PMMCS $r_c=8$ Å — Number density of all atoms ($r < 3.8$ Å)  |  723 K'
    '\n'
    r'75SiO$_2\cdot$(15$-x$)Na$_2$O$\cdot x$K$_2$O$\cdot$10CaO',
    fontsize=9.5
)

fig.savefig(HERE / 'fig_number_density_vargheese.pdf', dpi=300, bbox_inches='tight')
fig.savefig(HERE / 'fig_number_density_vargheese.png', dpi=150, bbox_inches='tight')
print(f"\nGuardado: {HERE / 'fig_number_density_vargheese.pdf'}")
plt.close(fig)
