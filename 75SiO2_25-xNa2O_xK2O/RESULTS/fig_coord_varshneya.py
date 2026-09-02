"""
fig_coord_varshneya.py
Reproducción de Figs. 3 y 4 de Tandia et al., J. Non-Cryst. Solids 358 (2012) 316.
Sistema: 75SiO2·(25-x)Na2O·xK2O·10CaO, PMMCS rc=8 Å, 723 K.

Tipos: Si=1, O=2, Ca=3, Na=4, K=5
Cutoff uniforme: 3.4 Å para todas las especies (Na-O y K-O).

Fig. 3 (2 paneles): distribución de número de coordinación O alrededor de Na y K
  - Panel a: x=6  (K2O=6%, Na2O=9%) — análogo a [Na2O]=10% en Tandia2012
  - Panel b: x=12 (K2O=12%, Na2O=3%) — análogo a [Na2O]=18% en Tandia2012
  - Tres series: Na-O (AM), K-O (AM), K-O (IEX1)

Fig. 4 (2 paneles):
  - Panel a: coordinación media K-O y Na-O vs x para vidrios AM (todos x)
  - Panel b: distribución K-O en:
      · AM x=15 (extremo puro K)
      · IEX1 xt=6 (intercambio parcial)
      · IEX1 xt=12 (intercambio parcial)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from collections import defaultdict

HERE      = Path(__file__).parent
BASE      = HERE.parent
AM_DIR    = BASE / "STAGE1_MELTQUENCH" / "data" / "asmelted_723K"
IEX1_DIR  = BASE / "STAGE3_IEX_PROTO1" / "data" / "iox1_723K"

R_CUT_NaO    = 3.40   # Å — primer mínimo g(Na-O) en AM
R_CUT_KO_AM  = 3.87   # Å — primer mínimo g(K-O) en AM
R_CUT_KO_IEX = 3.40   # Å — K en IEX1 ocupa sitios Na → mismo cutoff que Na-O

X_AM   = [0, 1, 3, 6, 9, 12, 15]
X_IEX1 = [1, 3, 6, 9, 12, 15]


# ── Parser ────────────────────────────────────────────────────────────────────

def read_data(path):
    """Devuelve L (array [3]), pos_by_type {type_int: Nx3 array}."""
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
                        t = int(p[1])
                        x, y, z = float(p[3]), float(p[4]), float(p[5])
                        atoms.append((t, x, y, z))
                    except ValueError:
                        pass
    L = np.array([box['x'], box['y'], box['z']])
    atoms = np.array(atoms)
    pos = {}
    for t in np.unique(atoms[:, 0]).astype(int):
        mask = atoms[:, 0] == t
        pos[t] = atoms[mask, 1:4]
    return L, pos


# ── Números de coordinación ───────────────────────────────────────────────────

def coord_numbers(pos_center, pos_target, L, r_cut, batch=200):
    """Número de átomos target dentro de r_cut de cada átomo center (PBC min-image)."""
    r2 = r_cut ** 2
    result = []
    for i in range(0, len(pos_center), batch):
        pc = pos_center[i:i + batch]        # (B, 3)
        dr = pc[:, None, :] - pos_target[None, :, :]  # (B, N_target, 3)
        dr -= np.round(dr / L) * L
        d2 = (dr ** 2).sum(axis=2)          # (B, N_target)
        result.extend((d2 < r2).sum(axis=1).tolist())
    return np.array(result, dtype=int)


def collect_coord(path, species, r_cut, TYPE_O=2):
    """Calcula array de coord. numbers de `species` (tipo) hacia O."""
    L, pos = read_data(path)
    if species not in pos or TYPE_O not in pos:
        return np.array([], dtype=int)
    return coord_numbers(pos[species], pos[TYPE_O], L, r_cut)


def collect_all_reps(data_fn, x_val, species, r_cut, reps=(1, 2, 3)):
    """Acumula coord. numbers de todas las réplicas de la composición x_val."""
    all_cn = []
    for r in reps:
        p = data_fn(x_val, r)
        if p.exists():
            cn = collect_coord(p, species, r_cut)
            all_cn.extend(cn.tolist())
    return np.array(all_cn, dtype=int)


# ── Distribuciones (histograma normalizado) ──────────────────────────────────

def cn_distribution(cn_array, bins=None):
    if bins is None:
        bins = np.arange(cn_array.min(), cn_array.max() + 2)
    counts, edges = np.histogram(cn_array, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    prob = counts / counts.sum() * 100
    return centers, prob


# ── Rutas de archivos ─────────────────────────────────────────────────────────

def am_path(x, r):
    return AM_DIR / f"AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data"


def iex1_path(x, r):
    return IEX1_DIR / f"IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data"


# ── Colores y estilos ─────────────────────────────────────────────────────────

C_NaO_AM  = '#1a6bb0'   # azul  — Na-O as-melted
C_KO_AM   = '#c0392b'   # rojo  — K-O as-melted
C_KO_IEX1 = '#27ae60'  # verde — K-O IEX1

KW_LINE = dict(lw=1.5, ms=5.5)
MARKERS = {'Na-O AM': 'D', 'K-O AM': 's', 'K-O IEX': '^'}

TYPE_Na, TYPE_K = 4, 5

# ──────────────────────────────────────────────────────────────────────────────
# FIG 3: distribución de coord. para x=6 y x=12
# ──────────────────────────────────────────────────────────────────────────────

print("Calculando Fig. 3 (distribuciones de coordinación)...")

fig3_data = {}   # {x: {'NaO_AM': cn_arr, 'KO_AM': cn_arr, 'KO_IEX1': cn_arr}}

for x in [6, 12]:
    fig3_data[x] = {
        'NaO_AM':  collect_all_reps(am_path,   x, TYPE_Na, R_CUT_NaO),
        'KO_AM':   collect_all_reps(am_path,   x, TYPE_K,  R_CUT_KO_AM),
        'KO_IEX1': collect_all_reps(iex1_path, x, TYPE_K,  R_CUT_KO_IEX),
    }
    for key, arr in fig3_data[x].items():
        if len(arr) > 0:
            print(f"  x={x}, {key}: N={len(arr)}, mean={arr.mean():.2f}±{arr.std():.2f}")

fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

labels = ['(a)', '(b)']
x_vals = [6, 12]
subtitles = [r'$x=6$ (K$_2$O=6%, Na$_2$O=9%)', r'$x=12$ (K$_2$O=12%, Na$_2$O=3%)']

for ax, x, label, subtitle in zip(axes3, x_vals, labels, subtitles):
    d = fig3_data[x]
    bins = np.arange(0, 20)

    # Na-O (AM)
    if len(d['NaO_AM']) > 0:
        c, p = cn_distribution(d['NaO_AM'], bins)
        ax.plot(c, p, color=C_NaO_AM, marker='D', label=r'Na–O (AM)', **KW_LINE)

    # K-O (AM)
    if len(d['KO_AM']) > 0:
        c, p = cn_distribution(d['KO_AM'], bins)
        ax.plot(c, p, color=C_KO_AM, marker='s', label=r'K–O (AM)', **KW_LINE)

    # K-O (IEX1)
    if len(d['KO_IEX1']) > 0:
        c, p = cn_distribution(d['KO_IEX1'], bins)
        ax.plot(c, p, color=C_KO_IEX1, marker='^', label=r'K–O (IEX1)', **KW_LINE)

    ax.set_xlabel('Coordination Number', fontsize=10)
    ax.set_ylabel('Percent of Alkali Ions (%)', fontsize=10)
    ax.set_xlim(0, 16)
    ax.set_title(f'{label}  {subtitle}', fontsize=9.5)
    ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='#cccccc')
    ax.tick_params(labelsize=8.5)
    ax.grid(axis='y', lw=0.3, color='#dddddd', zorder=0)

fig3.suptitle(
    r'PMMCS $r_c=8$ Å — O coordination number distribution  |  723 K'
    '\n'
    r'Cutoffs: Na–O = 3.40 Å, K–O (AM) = 3.87 Å, K–O (IEX1) = 3.40 Å'
    '\n'
    r'75SiO$_2\cdot$(15$-x$)Na$_2$O$\cdot x$K$_2$O$\cdot$10CaO',
    fontsize=9.5
)
fig3.tight_layout()
fig3.savefig(HERE / 'fig_coord_varshneya_fig3.pdf', dpi=300, bbox_inches='tight')
fig3.savefig(HERE / 'fig_coord_varshneya_fig3.png', dpi=150, bbox_inches='tight')
print(f"Guardado: fig_coord_varshneya_fig3.pdf")
plt.close(fig3)


# ──────────────────────────────────────────────────────────────────────────────
# FIG 4: (a) coordinación media vs x; (b) distribución K-O en AM puro y IEX1
# ──────────────────────────────────────────────────────────────────────────────

print("\nCalculando Fig. 4 (coordinación media y distribución K-O extremos)...")

# Panel a: media de K-O y Na-O en AM para todos los x
mean_KO_am  = []
mean_NaO_am = []

for x in X_AM:
    ko = collect_all_reps(am_path, x, TYPE_K,  R_CUT_KO_AM)
    na = collect_all_reps(am_path, x, TYPE_Na, R_CUT_NaO)
    mean_KO_am.append(ko.mean() if len(ko) > 0 else np.nan)
    mean_NaO_am.append(na.mean() if len(na) > 0 else np.nan)

mean_KO_am  = np.array(mean_KO_am)
mean_NaO_am = np.array(mean_NaO_am)

# K-O en IEX1 (medias para eje a)
mean_KO_iex1 = []
for x in X_IEX1:
    ko = collect_all_reps(iex1_path, x, TYPE_K, R_CUT_KO_IEX)
    mean_KO_iex1.append(ko.mean() if len(ko) > 0 else np.nan)
mean_KO_iex1 = np.array(mean_KO_iex1)

# Panel b: distribución K-O en AM x=15 y IEX1 xt=6, xt=12
ko_am15  = collect_all_reps(am_path,   15, TYPE_K, R_CUT_KO_AM)
ko_ix6   = collect_all_reps(iex1_path,  6, TYPE_K, R_CUT_KO_IEX)
ko_ix12  = collect_all_reps(iex1_path, 12, TYPE_K, R_CUT_KO_IEX)

fig4, axes4 = plt.subplots(1, 2, figsize=(10, 5))

# ── Panel 4a ──────────────────────────────────────────────────────────────────
ax4a = axes4[0]
Na2O_at_x = [15 - x for x in X_AM]    # [Na2O] mol% para eje del paper
Na2O_iex1  = [15 - x for x in X_IEX1]

ax4a.plot(Na2O_at_x, mean_KO_am,  color=C_KO_AM,   marker='s', label='K–O (AM)',   **KW_LINE)
ax4a.plot(Na2O_iex1, mean_KO_iex1, color=C_KO_IEX1, marker='^', label='K–O (IEX1)', **KW_LINE)
ax4a.plot(Na2O_at_x, mean_NaO_am, color=C_NaO_AM,  marker='D', label='Na–O (AM)',  **KW_LINE)

ax4a.set_xlabel(r'[Na$_2$O] en el vidrio AM (mol%)', fontsize=10)
ax4a.set_ylabel('Average Coordination', fontsize=10)
ax4a.set_xlim(-1, 16)
ax4a.set_xticks([0, 3, 6, 9, 12, 15])
ax4a.set_title('(a)', fontsize=10, loc='left')
ax4a.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='#cccccc')
ax4a.tick_params(labelsize=8.5)
ax4a.grid(axis='y', lw=0.3, color='#dddddd', zorder=0)

# ── Panel 4b ──────────────────────────────────────────────────────────────────
ax4b = axes4[1]
bins = np.arange(0, 20)

C_am15 = '#c0392b'   # rojo — AM puro K
C_ix6  = '#8e44ad'  # morado — IEX1 xt=6
C_ix12 = '#27ae60'  # verde — IEX1 xt=12

for arr, color, label, marker in [
    (ko_am15, C_am15, 'AM puro K ($x=15$)',         's'),
    (ko_ix6,  C_ix6,  r'IEX1 $x_t=6$',             '^'),
    (ko_ix12, C_ix12, r'IEX1 $x_t=12$',            'D'),
]:
    if len(arr) > 0:
        c, p = cn_distribution(arr, bins)
        ax4b.plot(c, p, color=color, marker=marker, label=label, **KW_LINE)

ax4b.set_xlabel('Coordination Number (K–O)', fontsize=10)
ax4b.set_ylabel('Percent of Potassium Ions (%)', fontsize=10)
ax4b.set_xlim(0, 16)
ax4b.set_title('(b)', fontsize=10, loc='left')
ax4b.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='#cccccc')
ax4b.tick_params(labelsize=8.5)
ax4b.grid(axis='y', lw=0.3, color='#dddddd', zorder=0)

fig4.suptitle(
    r'PMMCS $r_c=8$ Å — K–O and Na–O coordination  |  723 K'
    '\n'
    r'Cutoffs: Na–O = 3.40 Å, K–O (AM) = 3.87 Å, K–O (IEX1) = 3.40 Å'
    '\n'
    r'75SiO$_2\cdot$(15$-x$)Na$_2$O$\cdot x$K$_2$O$\cdot$10CaO',
    fontsize=9.5
)
fig4.tight_layout()
fig4.savefig(HERE / 'fig_coord_varshneya_fig4.pdf', dpi=300, bbox_inches='tight')
fig4.savefig(HERE / 'fig_coord_varshneya_fig4.png', dpi=150, bbox_inches='tight')
print(f"Guardado: fig_coord_varshneya_fig4.pdf")
plt.close(fig4)
