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
        "legend.fontsize": 11,
        "xtick.labelsize": fnt_size,
        "ytick.labelsize": fnt_size,
    }
)
plt.rcParams["axes.linewidth"] = 1.25


BASE_DIR = Path(__file__).resolve().parent.parent / "YOUNG_MODULUS_STRAIN_STRESS"
AXIS_CONFIG = {
    "x": {"length_col": "Lx", "stress_col": "Pxx", "color": "#1f77b4"},
    "y": {"length_col": "Ly", "stress_col": "Pyy", "color": "#d62728"},
    "z": {"length_col": "Lz", "stress_col": "Pzz", "color": "#2ca02c"},
}
X_VALUES = [0, 1, 2, 3, 4, 5, 6, 9, 12, 15]
PID_VALUES = [0, 1, 2, 3, 4, 5]

# LAMMPS units metal -> pressure in bar. 1 bar = 1e-4 GPa.
BAR_TO_GPA = 1.0e-4
FIT_EPS_MAX = 0.02


def compute_strain_stress(log_path: Path, length_col: str, stress_col: str):
    log = lammps_logfile.File(str(log_path))
    length = np.asarray(log.get(length_col), dtype=float)
    stress_bar = np.asarray(log.get(stress_col), dtype=float)

    l0 = float(length[0])
    strain = (length - l0) / l0
    # Sign convention: tensile stress positive.
    stress_gpa = -stress_bar * BAR_TO_GPA
    return strain, stress_gpa, l0


def compute_all_strains(log_path: Path):
    log = lammps_logfile.File(str(log_path))
    lx = np.asarray(log.get("Lx"), dtype=float)
    ly = np.asarray(log.get("Ly"), dtype=float)
    lz = np.asarray(log.get("Lz"), dtype=float)
    ex = (lx - lx[0]) / lx[0]
    ey = (ly - ly[0]) / ly[0]
    ez = (lz - lz[0]) / lz[0]
    return {"x": ex, "y": ey, "z": ez}


def linear_fit_modulus(strain, stress_gpa, eps_max):
    mask = (strain >= 0.0) & (strain <= eps_max)
    x = strain[mask]
    y = stress_gpa[mask]
    if x.size < 2:
        raise ValueError(f"No hay suficientes puntos para ajustar hasta strain={eps_max}.")

    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), x, y


def linear_fit_slope(x_values, y_values, x_max):
    mask = (x_values >= 0.0) & (x_values <= x_max)
    x = x_values[mask]
    y = y_values[mask]
    if x.size < 2:
        raise ValueError(f"No hay suficientes puntos para ajustar hasta strain={x_max}.")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def mean_std_finite(values):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if np.any(finite):
        return float(np.mean(values[finite])), float(np.std(values[finite]))
    return np.nan, np.nan


def linear_fit_with_band(ax, x_arr, y_mean, y_std, color, ylabel, title):
    ax.plot(x_arr, y_mean, "-o", lw=2.0, ms=5.0, color=color, label="Average")
    ax.fill_between(
        x_arr,
        y_mean - y_std,
        y_mean + y_std,
        color=color,
        alpha=0.22,
        linewidth=0.0,
        label=r"$\pm \sigma$",
    )

    finite = np.isfinite(y_mean)
    if np.sum(finite) >= 2:
        slope, intercept = np.polyfit(x_arr[finite], y_mean[finite], 1)
        y_fit = slope * x_arr + intercept
        ax.plot(x_arr, y_fit, "--", lw=2.0, color="#ff7f0e", label=rf"Fit: y={slope:.3f}x+{intercept:.3f}")

    ax.grid(True, alpha=0.35)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")


def main():
    fig_ss, axes_grid = plt.subplots(2, 5, figsize=(16, 8), constrained_layout=True)
    axes_flat = axes_grid.ravel()

    young_mean = []
    young_std = []
    poisson_mean = []
    poisson_std = []
    bulk_mean = []
    bulk_std = []
    shear_mean = []
    shear_std = []

    for i, x_val in enumerate(X_VALUES):
        ax_panel = axes_flat[i]
        young_by_axis_mean = {}
        poisson_by_axis_mean = {}

        for axis_name, cfg in AXIS_CONFIG.items():
            young_pid = []
            poisson_pid = []
            strain_pid = []
            stress_pid = []

            for pid in PID_VALUES:
                filename = f"input_{axis_name}_{pid}_T300_x{x_val}.lammps"
                log_path = BASE_DIR / filename
                if not log_path.exists():
                    raise FileNotFoundError(f"No existe el archivo: {log_path}")

                strain, stress_gpa, _ = compute_strain_stress(log_path, cfg["length_col"], cfg["stress_col"])
                young, _, _, _ = linear_fit_modulus(strain, stress_gpa, FIT_EPS_MAX)
                young_pid.append(young)
                strain_pid.append(strain)
                stress_pid.append(stress_gpa)

                strains_all = compute_all_strains(log_path)
                axial = strains_all[axis_name]
                transverse_axes = [ax for ax in ["x", "y", "z"] if ax != axis_name]
                nu_trans = []
                for t_axis in transverse_axes:
                    slope_trans, _ = linear_fit_slope(axial, strains_all[t_axis], FIT_EPS_MAX)
                    nu_trans.append(-slope_trans)
                poisson_pid.append(float(np.mean(nu_trans)))

            young_by_axis_mean[axis_name] = float(np.mean(young_pid))
            poisson_by_axis_mean[axis_name] = float(np.mean(poisson_pid))

            first_shape = strain_pid[0].shape
            if all(arr.shape == first_shape for arr in strain_pid) and all(arr.shape == first_shape for arr in stress_pid):
                strain_avg = np.mean(np.stack(strain_pid, axis=0), axis=0)
                stress_avg = np.mean(np.stack(stress_pid, axis=0), axis=0)
                ax_panel.plot(
                    strain_avg,
                    stress_avg,
                    "-",
                    lw=1.25,
                    color=cfg["color"],
                    alpha=0.9,
                    label=axis_name.upper(),
                )
            else:
                for strain_i, stress_i in zip(strain_pid, stress_pid):
                    ax_panel.plot(
                        strain_i,
                        stress_i,
                        "-",
                        lw=1.0,
                        color=cfg["color"],
                        alpha=0.25,
                    )
                ax_panel.plot([], [], "-", lw=1.25, color=cfg["color"], alpha=0.9, label=axis_name.upper())

        young_values = np.asarray([young_by_axis_mean[ax] for ax in ["x", "y", "z"]], dtype=float)
        poisson_values = np.asarray([poisson_by_axis_mean[ax] for ax in ["x", "y", "z"]], dtype=float)

        young_mean.append(float(np.mean(young_values)))
        young_std.append(float(np.std(young_values)))
        poisson_mean.append(float(np.mean(poisson_values)))
        poisson_std.append(float(np.std(poisson_values)))

        bulk_values = []
        shear_values = []
        for axis_name in ["x", "y", "z"]:
            e_i = young_by_axis_mean[axis_name]
            nu_i = poisson_by_axis_mean[axis_name]

            den_k = 3.0 * (1.0 - 2.0 * nu_i)
            den_g = 2.0 * (1.0 + nu_i)
            k_i = np.nan if abs(den_k) < 1.0e-10 else e_i / den_k
            g_i = np.nan if abs(den_g) < 1.0e-10 else e_i / den_g
            bulk_values.append(k_i)
            shear_values.append(g_i)

        k_mean, k_std = mean_std_finite(bulk_values)
        g_mean, g_std = mean_std_finite(shear_values)
        bulk_mean.append(k_mean)
        bulk_std.append(k_std)
        shear_mean.append(g_mean)
        shear_std.append(g_std)

        ax_panel.grid(True, alpha=0.35)
        ax_panel.set_title(f"x={x_val}")
        ax_panel.set_xlabel(r"Strain $\varepsilon$")
        if i % 10 == 0:
            ax_panel.set_ylabel(r"Stress $\sigma$ (GPa)")
        if i == 0:
            ax_panel.legend(frameon=False, loc="best")

    for j in range(len(X_VALUES), len(axes_flat)):
        axes_flat[j].axis("off")

    fig_ss.suptitle(rf"Stress-Strain por composicion ($\varepsilon \leq {FIT_EPS_MAX:.2f}$ para ajuste de E)")
    fig_ss.savefig("estimate_young_strain_stress_grid_2x10.pdf", bbox_inches="tight")
    fig_ss.savefig("estimate_young_strain_stress_grid_2x10.png", dpi=400, bbox_inches="tight")

    x_arr = np.asarray(X_VALUES, dtype=float)
    young_mean = np.asarray(young_mean, dtype=float)
    young_std = np.asarray(young_std, dtype=float)
    poisson_mean = np.asarray(poisson_mean, dtype=float)
    poisson_std = np.asarray(poisson_std, dtype=float)
    bulk_mean = np.asarray(bulk_mean, dtype=float)
    bulk_std = np.asarray(bulk_std, dtype=float)
    shear_mean = np.asarray(shear_mean, dtype=float)
    shear_std = np.asarray(shear_std, dtype=float)

    fig_props, axes_props = plt.subplots(2, 2, figsize=(11.2, 8.0), constrained_layout=True)
    linear_fit_with_band(
        ax=axes_props[0, 0],
        x_arr=x_arr,
        y_mean=young_mean,
        y_std=young_std,
        color="#1f77b4",
        ylabel="Young modulus (GPa)",
        title="Young",
    )
    linear_fit_with_band(
        ax=axes_props[0, 1],
        x_arr=x_arr,
        y_mean=bulk_mean,
        y_std=bulk_std,
        color="#2ca02c",
        ylabel="Bulk modulus (GPa)",
        title="Bulk",
    )
    linear_fit_with_band(
        ax=axes_props[1, 0],
        x_arr=x_arr,
        y_mean=shear_mean,
        y_std=shear_std,
        color="#9467bd",
        ylabel="Shear modulus (GPa)",
        title="Shear",
    )
    linear_fit_with_band(
        ax=axes_props[1, 1],
        x_arr=x_arr,
        y_mean=poisson_mean,
        y_std=poisson_std,
        color="#8c564b",
        ylabel="Poisson ratio",
        title="Poisson",
    )
    fig_props.savefig("estimate_elastic_properties_vs_x.pdf", bbox_inches="tight")
    fig_props.savefig("estimate_elastic_properties_vs_x.png", dpi=400, bbox_inches="tight")

    print(f"Young modulus usando ajuste lineal hasta strain <= {FIT_EPS_MAX:.2f}")
    for x_val, e_mean, e_std, k_mean, k_std, g_mean, g_std, nu_mean, nu_std in zip(
        X_VALUES,
        young_mean,
        young_std,
        bulk_mean,
        bulk_std,
        shear_mean,
        shear_std,
        poisson_mean,
        poisson_std,
    ):
        print(
            f"  x={x_val:>2} | "
            f"E={e_mean:.3f}±{e_std:.3f} GPa | "
            f"K={k_mean:.3f}±{k_std:.3f} GPa | "
            f"G={g_mean:.3f}±{g_std:.3f} GPa | "
            f"nu={nu_mean:.4f}±{nu_std:.4f}"
        )

    plt.show()


if __name__ == "__main__":
    main()
