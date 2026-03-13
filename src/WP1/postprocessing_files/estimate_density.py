import re
from pathlib import Path

import lammps_logfile
import matplotlib.pyplot as plt
import numpy as np


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
X_VALUES = [0, 3, 6, 9, 12, 15]
EXP_DENSITY = np.array([2.48211, 2.481, 2.48284, 2.47479, 2.47525, 2.45893], dtype=float)
LAST_N_POINTS = 20


def parse_composition_data(log_path: Path):
    group_counts = {}
    masses = {}

    group_pattern = re.compile(r"^\s*(\d+)\s+atoms in group\s+([A-Za-z0-9_]+)\s*$")
    mass_pattern = re.compile(r"^\s*mass\s+\d+\s+([0-9]*\.?[0-9]+)\s+#\s*([A-Za-z0-9_]+)\s*$")

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            group_match = group_pattern.match(line)
            if group_match:
                count, species = group_match.groups()
                group_counts[species] = int(count)
                continue

            mass_match = mass_pattern.match(line)
            if mass_match:
                mass_value, species = mass_match.groups()
                masses[species] = float(mass_value)

    needed_species = ["Si", "O", "Ca", "Na", "K"]
    missing = [sp for sp in needed_species if sp not in group_counts or sp not in masses]
    if missing:
        raise ValueError(f"Faltan datos de composición en {log_path.name}: {missing}")

    return group_counts, masses


def get_density_stats(log_path: Path, n_last: int):
    log = lammps_logfile.File(str(log_path))
    density = np.asarray(log.get("Density"), dtype=float)
    if density.size < n_last:
        raise ValueError(f"{log_path.name} tiene {density.size} puntos; se requieren al menos {n_last}.")

    tail = density[-n_last:]
    return float(np.mean(tail)), float(np.std(tail, ddof=1)), tail


def get_molar_mass_from_log(log_path: Path):
    group_counts, masses = parse_composition_data(log_path)

    n_si = group_counts["Si"]
    n_ca = group_counts["Ca"]
    n_na = group_counts["Na"]
    n_k = group_counts["K"]
    oxide_units = n_si + n_ca + 0.5 * n_na + 0.5 * n_k

    total_mass = sum(group_counts[sp] * masses[sp] for sp in ["Si", "O", "Ca", "Na", "K"])
    molar_mass = total_mass / oxide_units
    return float(molar_mass)


def main():
    density_mean = []
    density_std = []
    molar_mass_values = []
    vm_mean = []
    vm_std = []
    density_rel_error = []
    vm_rel_error = []

    for x_val, exp_rho in zip(X_VALUES, EXP_DENSITY):
        log_path = Path(str(BASE_PATH).format(x=x_val))
        rho_mean, rho_std, rho_tail = get_density_stats(log_path, LAST_N_POINTS)
        molar_mass = get_molar_mass_from_log(log_path)

        vm_tail = molar_mass / rho_tail
        vm_avg = float(np.mean(vm_tail))
        vm_err = abs(molar_mass / (rho_mean**2)) * rho_std

        density_mean.append(rho_mean)
        density_std.append(rho_std)
        molar_mass_values.append(molar_mass)
        vm_mean.append(vm_avg)
        vm_std.append(vm_err)

        rel_error = (rho_mean - exp_rho) * 100.0 / exp_rho
        density_rel_error.append(rel_error)
        print(
            f"x={x_val:>2d} | rho = {rho_mean:.5f} +- {rho_std:.5f} g/cm^3 | "
            f"err_rel = {rel_error:+.3f}% | M = {molar_mass:.4f} g/mol | "
            f"V_m = {vm_avg:.4f} +- {vm_err:.4f} cm^3/mol"
        )

    x_arr = np.asarray(X_VALUES, dtype=float)
    rho_mean_arr = np.asarray(density_mean, dtype=float)
    rho_std_arr = np.asarray(density_std, dtype=float)
    vm_mean_arr = np.asarray(vm_mean, dtype=float)
    vm_std_arr = np.asarray(vm_std, dtype=float)
    molar_mass_arr = np.asarray(molar_mass_values, dtype=float)
    vm_exp_arr = molar_mass_arr / EXP_DENSITY
    density_rel_error_arr = np.asarray(density_rel_error, dtype=float)
    vm_rel_error_arr = (vm_mean_arr - vm_exp_arr) * 100.0 / vm_exp_arr

    fig1, ax1 = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    ax1.plot(x_arr, EXP_DENSITY, "o-", lw=1.8, ms=6, color="#111111", label="Experimental")
    ax1.plot(x_arr, rho_mean_arr, "s-", lw=2.0, ms=6, color="#1f77b4", label="MD")
    ax1.fill_between(
        x_arr,
        rho_mean_arr - rho_std_arr,
        rho_mean_arr + rho_std_arr,
        color="#1f77b4",
        alpha=0.25,
        linewidth=0,
    )
    ax1.set_xlabel(r"$x$ in $(15-x)$Na$_2$O-$x$K$_2$O")
    ax1.set_ylabel(r"$\rho$ (g/cm$^3$)")
    ax1.set_xticks(X_VALUES)
    ax1.grid(True, alpha=0.4)
    ax1.legend(frameon=False)
    fig1.savefig("estimate_density_vs_x.pdf", bbox_inches="tight")
    fig1.savefig("estimate_density_vs_x.png", dpi=400, bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    ax2.plot(x_arr, vm_exp_arr, "o-", lw=1.8, ms=6, color="#111111", label="Experimental")
    ax2.plot(x_arr, vm_mean_arr, "s-", lw=2.0, ms=6, color="#d62728", label="MD")
    # Franja de incertidumbre del volumen molar simulado (media ± desvio estandar).
    ax2.fill_between(
        x_arr,
        vm_mean_arr - vm_std_arr,
        vm_mean_arr + vm_std_arr,
        color="#d62728",
        alpha=0.25,
        linewidth=0,
    )
    ax2.set_xlabel(r"$x$ in $(15-x)$Na$_2$O-$x$K$_2$O")
    ax2.set_ylabel(r"$V_m$ (cm$^3$/mol)")
    ax2.set_xticks(X_VALUES)
    ax2.grid(True, alpha=0.4)
    ax2.legend(frameon=False)
    fig2.savefig("estimate_molar_volume_vs_x.pdf", bbox_inches="tight")
    fig2.savefig("estimate_molar_volume_vs_x.png", dpi=400, bbox_inches="tight")

    fig3, ax3 = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    ax3.plot(x_arr, density_rel_error_arr, "o-", lw=2.0, ms=6, color="#1f77b4", label=r"Relative error $\rho$")
    ax3.plot(x_arr, vm_rel_error_arr, "s-", lw=2.0, ms=6, color="#d62728", label=r"Relative error $V_m$")
    ax3.axhline(0.0, color="#444444", lw=1.2, alpha=0.7)
    ax3.set_xlabel(r"$x$ in $(15-x)$Na$_2$O-$x$K$_2$O")
    ax3.set_ylabel("Relative error (%)")
    ax3.set_xticks(X_VALUES)
    ax3.grid(True, alpha=0.4)
    ax3.legend(frameon=False)
    fig3.savefig("estimate_relative_errors_vs_x.pdf", bbox_inches="tight")
    fig3.savefig("estimate_relative_errors_vs_x.png", dpi=400, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
