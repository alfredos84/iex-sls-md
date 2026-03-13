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
SKIP_POINTS = 10

# Asuncion: el set de baja T se toma como T < 800 K.
LOW_TEMP_MAX = 1000.0
HIGH_TEMP_MIN = 2600.0


def linear_fit(x_data, y_data):
    if x_data.size < 2:
        raise ValueError("No hay suficientes puntos para ajuste lineal.")
    slope, intercept = np.polyfit(x_data, y_data, 1)
    return float(slope), float(intercept)


def cooling_mask(temperature, tol=1e-8):
    dtemp = np.gradient(temperature)
    return dtemp < -tol


def intersection_temperature(m1, b1, m2, b2):
    denom = m1 - m2
    if abs(denom) < 1e-14:
        return np.nan
    return (b2 - b1) / denom


fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
axes = axes.ravel()
tg_values = []

for ax, x_value in zip(axes, X_VALUES):
    log = lammps_logfile.File(str(BASE_PATH).format(x=x_value))

    temperature = np.asarray(log.get("Temp"), dtype=float)
    temperature = temperature[::10]
    density = np.asarray(log.get("Density"), dtype=float)
    density = density[::10]
    specific_volume = 1.0 / density

    idx_all = np.arange(temperature.size)
    mask_cooling = cooling_mask(temperature)

    low_mask = (temperature < LOW_TEMP_MAX) & mask_cooling
    high_mask = (temperature > HIGH_TEMP_MIN) & (idx_all >= SKIP_POINTS) & mask_cooling

    # Fallback: si quedan pocos puntos, relaja el filtro de enfriamiento.
    if np.count_nonzero(low_mask) < 5:
        low_mask = temperature < LOW_TEMP_MAX
    if np.count_nonzero(high_mask) < 5:
        high_mask = (temperature > HIGH_TEMP_MIN) & (idx_all >= SKIP_POINTS)

    temp_low = temperature[low_mask]
    v_low = specific_volume[low_mask]
    temp_high = temperature[high_mask]
    v_high = specific_volume[high_mask]

    m_low, b_low = linear_fit(temp_low, v_low)
    m_high, b_high = linear_fit(temp_high, v_high)
    tg = intersection_temperature(m_low, b_low, m_high, b_high)
    tg_values.append(tg)

    t_line_low = np.linspace(np.min(temp_low), np.max(temp_low), 100)
    t_line_high = np.linspace(np.min(temp_high), np.max(temp_high), 100)
    fit_low = m_low * t_line_low + b_low
    fit_high = m_high * t_line_high + b_high

    ax.scatter(temperature[SKIP_POINTS:], specific_volume[SKIP_POINTS:], marker=".", s=6, color="#7f7f7f")
    ax.scatter(temp_low, v_low, marker=".", s=9, color="#1f77b4", alpha=0.75)
    ax.scatter(temp_high, v_high, marker=".", s=9, color="#d62728", alpha=0.75)
    ax.plot(t_line_low, fit_low, "-", lw=2.0, color="#1f77b4")
    ax.plot(t_line_high, fit_high, "-", lw=2.0, color="#d62728")

    ax.grid(True, alpha=0.45)
    ax.set_box_aspect(1)
    ax.text(
        0.04,
        0.96,
        rf"$x={x_value}$"
        "\n"
        rf"$T_g={tg:.1f}\ \mathrm{{K}}$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=14,
    )
    ax.set_xlabel(r"$T$ (K)")
    ax.set_ylabel(r"$v_{\mathrm{spec}}$ (cm$^3$/g)")
    ax.set_xlim([200, 3200])
    ax.set_ylim([0.38, 0.52])
    ax.set_xticks(range(200, 3201, 1000))

plt.savefig("estimate_Tg_fits.pdf", bbox_inches="tight")
plt.savefig("estimate_Tg_fits.png", dpi=400, bbox_inches="tight")

tg_array = np.asarray(tg_values, dtype=float)
fig2, ax2 = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
ax2.plot(X_VALUES, tg_array, "o-", color="#2ca02c", lw=2.0, ms=6)
ax2.set_xlabel(r"$x$ $(15-x)$Na$_2$O-$x$K$_2$O")
ax2.set_ylabel(r"$T_g$ (K)")
ax2.set_xticks(X_VALUES)
ax2.grid(True, alpha=0.4)

fig2.savefig("estimate_Tg_vs_x.pdf", bbox_inches="tight")
fig2.savefig("estimate_Tg_vs_x.png", dpi=400, bbox_inches="tight")

for x_val, tg in zip(X_VALUES, tg_array):
    print(f"x={x_val:>2d} | Tg={tg:.2f} K")

plt.show()
