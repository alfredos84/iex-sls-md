import lammps_logfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


fnt_size = 13
plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "serif",
        "axes.labelsize": fnt_size,
        "legend.fontsize": 12,
        "xtick.labelsize": fnt_size,
        "ytick.labelsize": fnt_size,
    }
)
plt.rcParams["axes.linewidth"] = 1.25


BASE_DIR = Path(__file__).resolve().parent.parent / "MELTQUENCH"
BASE_PATH = BASE_DIR / "MeltQuenchGlass_x{x}_T300.lammps"
X_VALUES = [0,1,2,3,4,5,6,9]
SKIP_POINTS = 50
EXP_DENSITY=[2.48211, 2.46275, 2.46118, 2.481, 2.45889, 2.46098, 2.48284, 2.47479] #, 2.47525, 2.45893]

fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
axes = axes.ravel()

for ax, x_value, exp_dens in zip(axes, X_VALUES, EXP_DENSITY):
    log = lammps_logfile.File(str(BASE_PATH).format(x=x_value))

    temperature = np.asarray(log.get("Temp"))
    density = np.asarray(log.get("Density"))
    specific_volume = 1.0 / density
    final_density = density[-1]

    ax.scatter(
        temperature[SKIP_POINTS:],
        specific_volume[SKIP_POINTS:],
        marker=".",
        s=6,
        label=rf"$({15-x_value})Na_2O/({x_value})K_2O$",
    )
    ax.grid(True, alpha=0.45)
    ax.legend(loc="lower right", frameon=False)
    ax.set_box_aspect(1)
    ax.text(
        0.04,
        0.96,
        rf"$\rho_{{\mathrm{{final}}}}={final_density:.3f}\ \mathrm{{g/cm^3}}$"
        "\n"
        rf"$\varepsilon_{{\mathrm{{\rho}}}}={(final_density-exp_dens)*100/exp_dens:.3f}\ (\%)$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=16,
    )

    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(r"$v_{\mathrm{spec}}$ (cm$^3$/g)")
    ax.set_xlim([200,3200])
    ax.set_ylim([0.38,0.52])
    ax.set_xticks(range(200,3201,1000))

plt.savefig("estimate_Tg.pdf", bbox_inches="tight")
plt.savefig("estimate_Tg.png", dpi=400, bbox_inches="tight")
plt.show()
