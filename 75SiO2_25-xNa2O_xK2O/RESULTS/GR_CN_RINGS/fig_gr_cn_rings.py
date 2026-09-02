"""
fig_gr_cn_rings.py
Lee output de RINGS (gr/gr-p/gr-{sp1}_{sp2}.dat y rstat/rings-5.dat)
y genera figuras de g(r), CN(r) y distribución de anillos.

Formato gr-p files (FORMAT 623): r  g(r)  CN(r)
Formato rstat/rings-5.dat (FORMAT 613): ring_size  N_rings  frac_per_atom  max  min

Pares solicitados: Si-O, Si-Si, O-O, Na-O, K-O, Ca-O, K-K, Na-K, Na-Na
Protocolos: AM, IEX1, IEX2
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

HERE = Path(__file__).parent

# ── Definiciones ──────────────────────────────────────────────────────────────

PROTOCOLS = {
    'AM':   {'x_list': [0, 1, 3, 6, 9, 12, 15], 'label': 'As-melted',    'ls': '-'},
    'IEX1': {'x_list': [1, 3, 6, 9, 12, 15],    'label': 'IEX Proto1',   'ls': '--'},
    'IEX2': {'x_list': [0, 1, 3, 6, 9, 12],     'label': 'IEX Proto2',   'ls': ':'},
}

# Pares a graficar (en notación RINGS: etiquetas sin espacio)
PAIRS = [
    ('Si', 'O'),
    ('Si', 'Si'),
    ('O',  'O'),
    ('Na', 'O'),
    ('K',  'O'),
    ('Ca', 'O'),
    ('K',  'K'),
    ('Na', 'K'),
    ('Na', 'Na'),
    ('Si', 'Na'),
    ('Si', 'K'),
]

# Paleta de colores para x values
X_COLORS = {
    0:  '#1a1a2e', 1:  '#1a6bb0', 3:  '#27ae60',
    6:  '#e67e22', 9:  '#8e44ad', 12: '#c0392b', 15: '#2c3e50',
}
X_COLORS_IEX1 = X_COLORS
X_COLORS_IEX2 = X_COLORS

PROTO_COLORS = {'AM': '#1a6bb0', 'IEX1': '#c0392b', 'IEX2': '#27ae60'}


# ── Leer g(r) y CN(r) de RINGS ───────────────────────────────────────────────

def load_gr(proto, x, sp1, sp2):
    """
    Lee gr-p/gr-{sp1}_{sp2}.dat del directorio {proto}_x{x}.
    Retorna (r, gr, cn) o (None, None, None) si no existe.
    """
    # RINGS nombra los archivos con los labels exactos (incluyendo espacios en 2-char)
    # Probamos ambas combinaciones sp1_sp2 y sp2_sp1
    run_dir = HERE / f"{proto}_x{x}"
    for s1, s2 in [(sp1, sp2), (sp2, sp1)]:
        p = run_dir / "gr" / "gr-p" / f"gr-{s1}_{s2}.dat"
        if p.exists():
            try:
                data = np.loadtxt(p, comments='#')
                if data.ndim == 2 and data.shape[1] >= 3:
                    return data[:, 0], data[:, 1], data[:, 2]
            except Exception:
                pass
    return None, None, None


# ── Leer ring statistics ──────────────────────────────────────────────────────

N_SI = 3750   # número de Si en todos los sistemas (constante)

def load_rings(proto, x, n_max=20):
    """
    Lee rstat/rings-2.dat (King's rings, R2).
    Retorna (n_Si_array, prob_array) normalizado a densidad de probabilidad:
    sum(prob) = 1 sobre n_Si in [2..n_max].
    """
    p = HERE / f"{proto}_x{x}" / "rstat" / "rings-2.dat"
    if not p.exists():
        return None, None
    raw = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('@'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    sz   = int(float(parts[0]))
                    frac = float(parts[1])
                    n_si = sz // 2
                    if 2 <= n_si <= n_max:
                        raw[n_si] = frac
                except (ValueError, IndexError):
                    pass
    if not raw:
        return None, None
    sizes = np.arange(2, n_max + 1, dtype=float)
    counts = np.array([raw.get(n, 0.0) for n in range(2, n_max + 1)])
    total = counts.sum()
    prob = counts / total if total > 0 else counts
    return sizes, prob


# ── Figuras g(r) y CN(r) ─────────────────────────────────────────────────────

def fig_gr_cn_pair(sp1, sp2):
    """
    Una figura por par, con 3 filas (AM, IEX1, IEX2) × 2 columnas (g(r), CN(r)).
    Cada curva es un valor de x.
    """
    pair_label = f"{sp1}–{sp2}"
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharey=False)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.08,
                        hspace=0.38, wspace=0.28)

    for row, proto in enumerate(['AM', 'IEX1', 'IEX2']):
        ax_gr = axes[row, 0]
        ax_cn = axes[row, 1]
        proto_info = PROTOCOLS[proto]

        any_data = False
        cn_max_all = 0.0
        for x in proto_info['x_list']:
            r, gr, cn = load_gr(proto, x, sp1, sp2)
            if r is None:
                continue
            c = X_COLORS.get(x, '#333333')
            ax_gr.plot(r, gr, lw=1.3, color=c, label=f'x={x}')
            ax_cn.plot(r, cn, lw=1.3, color=c, label=f'x={x}')
            any_data = True
            # CN máximo en rango visible [0, 8]
            mask = r <= 8.0
            if mask.any():
                cn_max_all = max(cn_max_all, cn[mask].max())

        for ax, ylabel in [(ax_gr, f'g(r) {pair_label}'),
                           (ax_cn, f'CN(r) {pair_label}')]:
            ax.set_xlabel('r (Å)', fontsize=9.5)
            ax.set_ylabel(ylabel, fontsize=9.5)
            ax.set_xlim(0, 8.0)
            ax.tick_params(labelsize=8.5)
            ax.grid(axis='y', lw=0.3, color='#dddddd', zorder=0)
            ax.set_title(f'{proto_info["label"]}', fontsize=9.5)
            if any_data:
                ax.legend(fontsize=7.5, ncol=2, frameon=True,
                          framealpha=0.85, edgecolor='#cccccc')
        # Escala CN: de 0 al máximo con 10% de margen
        if any_data and cn_max_all > 0:
            ax_cn.set_ylim(0, cn_max_all * 1.15)

    fname = f"fig_gr_cn_{sp1}{sp2}"
    fig.savefig(HERE / f"{fname}.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(HERE / f"{fname}.png", dpi=150, bbox_inches='tight')
    print(f"Guardado: {fname}.pdf")
    plt.close(fig)


# ── Figura de anillos ─────────────────────────────────────────────────────────

def fig_rings():
    """
    3 paneles (AM, IEX1, IEX2). Cada curva = un valor de x.
    Eje x = número de Si en el anillo (= tamaño RINGS / 2).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13, wspace=0.12)

    for ax, proto in zip(axes, ['AM', 'IEX1', 'IEX2']):
        proto_info = PROTOCOLS[proto]
        any_data = False
        for x in proto_info['x_list']:
            sizes, fracs = load_rings(proto, x)
            if sizes is None:
                continue
            c = X_COLORS.get(x, '#333333')
            ax.plot(sizes, fracs, marker='o', ms=4.5, lw=1.3,
                    color=c, label=f'x={x}', zorder=3)
            any_data = True

        ax.set_xlabel('Ring size', fontsize=10)
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_title(proto_info['label'], fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(axis='y', lw=0.3, color='#dddddd', zorder=0)
        ax.set_xlim(1.5, 20.5)
        ax.set_xticks(range(2, 21))
        ax.tick_params(axis='x', labelsize=7.5)
        if any_data:
            ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='#cccccc')

    fig.savefig(HERE / "fig_rings.pdf", dpi=300, bbox_inches='tight')
    fig.savefig(HERE / "fig_rings.png", dpi=150, bbox_inches='tight')
    print("Guardado: fig_rings.pdf")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for sp1, sp2 in PAIRS:
        fig_gr_cn_pair(sp1, sp2)
    fig_rings()
    print("Todas las figuras generadas.")
