"""Corte transversal (x-z, rebanada fina en y) de la indentacion en 4 momentos
criticos: antes de cargar, carga maxima, fin del hold, despues de descargar.
Datos extraidos directamente del indent.lammpstrj real en DEVANA (streaming,
sin usar OVITO/ASE por no estar instalados aca, pero el resultado visual es
equivalente a un corte de OVITO)."""
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

COLORS = {1: "#1a3a5c", 2: "#c0392b", 3: "#2ca0c4", 4: "#e08c2a", 5: "#8e44ad", 6: "#2c2c2c"}
LABELS = {1: "Si", 2: "O", 3: "Ca", 4: "Na", 5: "K", 6: "C (tip)"}
SIZES = {1: 4, 2: 3, 3: 5, 4: 5, 5: 5, 6: 2}

frames = [
    (20000, "Antes de cargar (t=20 ps)"),
    (120000, "Carga maxima (t=120 ps)"),
    (220000, "Fin del hold (t=220 ps)"),
    (320000, "Despues de descargar (t=320 ps)"),
]

fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

for ax, (step, title) in zip(axes, frames):
    fname = HERE / f"frame_{step}.txt"
    by_type = {t: ([], []) for t in COLORS}
    with open(fname) as f:
        header = f.readline()
        for line in f:
            aid, atype, x, z = line.split()
            atype = int(atype)
            by_type[atype][0].append(float(x))
            by_type[atype][1].append(float(z))
    for t in [1, 2, 3, 4, 5, 6]:
        xs, zs = by_type[t]
        if xs:
            ax.scatter(xs, zs, s=SIZES[t], c=COLORS[t], label=LABELS[t], linewidths=0, alpha=0.8)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel("x (Å)", fontsize=10)
    ax.set_xlim(0, 173.4)
    ax.set_ylim(-15, 180)
    ax.set_aspect("equal")
    ax.grid(lw=0.3, color="#dddddd", zorder=0)

axes[0].set_ylabel("z (Å)", fontsize=10)
axes[0].legend(fontsize=8, loc="upper right", frameon=True, framealpha=0.9, markerscale=2)
fig.suptitle("Corte transversal (|y-y_mid|<3.5 Å) — CaO x=0 r=1, geometría corregida (DEVANA)", fontsize=12)
fig.tight_layout()
fig.savefig(HERE / "fig_indent_snapshots.png", dpi=180)
fig.savefig(HERE / "fig_indent_snapshots.pdf")
print("Guardado: fig_indent_snapshots.png/.pdf")
