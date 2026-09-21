"""Curva carga-penetracion (P-h) de la indentacion piloto CaO x=0 r=1."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
EV_A_TO_NN = 1.602176634  # 1 eV/A = 1.602176634 nN

data = np.loadtxt(HERE / "tip_force_vs_disp.dat", comments="#")
step, fz, zpos = data[:, 0], data[:, 1], data[:, 2]

# Signo: compresion (fz negativo en la convencion de LAMMPS) -> carga positiva
F = -fz if np.median(fz[step > 1e5]) < 0 else fz
F_nN = F * EV_A_TO_NN

# Contacto: primer punto donde la fuerza supera el ruido de fondo
noise = np.median(np.abs(F[:10]))
i_contact = np.argmax(np.abs(F) > max(5 * noise, 1e-6))
z_contact = zpos[i_contact]
h = z_contact - zpos  # profundidad de penetracion (A), 0 en el contacto

loading = (step >= 20000) & (step <= 120000)
holding = (step >= 120000) & (step <= 220000)
unloading = (step >= 220000) & (step <= 320000)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot(h[loading], F_nN[loading], color="#1a3a5c", lw=1.8, label="Carga")
ax.plot(h[holding], F_nN[holding], color="#4cbf8f", lw=1.8, label="Hold (100 ps)")
ax.plot(h[unloading], F_nN[unloading], color="#c0392b", lw=1.8, label="Descarga")

ax.set_xlabel("Indentation depth, h (Å)", fontsize=11)
ax.set_ylabel("Load, P (nN)", fontsize=11)
ax.set_title("Nanoindentation P-h curve — CaO x=0 r=1 (AM)", fontsize=11)
ax.grid(axis="y", lw=0.35, color="#dddddd", zorder=0)
ax.tick_params(labelsize=9.5)
ax.legend(fontsize=9.5, frameon=True, framealpha=0.9, edgecolor="#cccccc")
fig.tight_layout()

fig.savefig(HERE / "fig_indentation_Ph.pdf")
fig.savefig(HERE / "fig_indentation_Ph.png", dpi=200)

Pmax = F_nN[loading].max()
hmax = h[loading][np.argmax(F_nN[loading])]
print(f"Pmax = {Pmax:.3f} nN")
print(f"hmax = {hmax:.3f} A")
print("(hf residual requiere el ajuste Oliver-Pharr de la pendiente de descarga,")
print(" no el ultimo punto crudo -- ver compute_hardness_OliverPharr.py)")
print("Figuras guardadas: fig_indentation_Ph.pdf / .png")
