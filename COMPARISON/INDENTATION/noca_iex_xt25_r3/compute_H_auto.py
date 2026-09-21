"""Oliver-Pharr automatico: ajusta a, m, hf juntos (sin eleccion visual de hf)
sobre tip_force_vs_disp.dat. Mismo modelo/tramo que compute_hardness_OliverPharr.py
de Pedone, para comparar contra el valor elegido a ojo."""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).parent
fname = sys.argv[1] if len(sys.argv) > 1 else "tip_force_vs_disp.dat"

# Geometria real de nuestra punta (create_slab_tip.in): r=40 A, h=52.5 A
TIP_R, TIP_H = 40.0, 52.5
theta_half_deg = np.degrees(np.arctan(TIP_R / TIP_H))  # semi-angulo real
epsilon_geom = 0.72
fit_first_frac = 0.35
min_fit_points = 10

EV_PER_ANG_TO_N = 1.602176634e-9
ANG_TO_M = 1e-10
PA_TO_GPA = 1e-9

data = np.loadtxt(HERE / fname, comments="#")
step, fz, zpos = data[:, 0], data[:, 1], data[:, 2]

i_absmax = np.argmax(np.abs(fz))
F = -fz if fz[i_absmax] < 0 else fz

# Mismo criterio de contacto que compute_hardness_OliverPharr.py (umbral
# relativo al 1% de Fmax, no un umbral fijo) para que hmax/H sean comparables.
n0 = max(10, int(0.10 * len(F)))
noise = np.median(np.abs(F[:n0]))
thr = max(5.0 * noise, 0.01 * F.max())
i_contact = np.argmax(np.abs(F) >= thr)
z_contact = zpos[i_contact]
h = z_contact - zpos

i_pmax = int(np.argmax(F))
Pmax, hmax = float(F[i_pmax]), float(h[i_pmax])

# Tramo de descarga tras el hold (salta la plateau)
start_unload = i_pmax
for k in range(i_pmax + 1, len(h)):
    if (h[i_pmax] - h[k]) > 0.01:
        start_unload = k
        break
h_un, P_un = h[start_unload:], F[start_unload:]
n_first = max(min_fit_points, int(fit_first_frac * len(h_un)))
h_fit, P_fit = h_un[:n_first], P_un[:n_first]

def P_model(hh, a, m, hf):
    return a * np.maximum(hh - hf, 0.0) ** m

p0 = [Pmax / (hmax * 0.5) ** 1.5, 1.5, hmax * 0.3]
bounds = ([0.0, 0.9, -50.0], [np.inf, 3.0, hmax * 0.95])
popt, _ = curve_fit(P_model, h_fit, P_fit, p0=p0, bounds=bounds, maxfev=40000)
a_fit, m_fit, hf_fit = [float(x) for x in popt]

S = a_fit * m_fit * max(hmax - hf_fit, 1e-9) ** (m_fit - 1.0)
hc = hmax - epsilon_geom * Pmax / S
theta = np.radians(theta_half_deg)
Ac = np.pi * (hc * ANG_TO_M) ** 2 * np.tan(theta) ** 2
H_GPa = (Pmax * EV_PER_ANG_TO_N / Ac) * PA_TO_GPA

print("=== Oliver-Pharr AUTOMATICO (hf ajustado, no elegido a ojo) ===")
print(f"Geometria punta: r={TIP_R} A, h={TIP_H} A -> semi-angulo = {theta_half_deg:.2f} deg")
print(f"Pmax = {Pmax:.3f} eV/A  ({Pmax*EV_PER_ANG_TO_N*1e9:.3f} nN)")
print(f"hmax = {hmax:.3f} A")
print(f"Ajuste: a={a_fit:.4e}  m={m_fit:.4f}  hf={hf_fit:.4f} A")
print(f"S (rigidez de contacto) = {S:.4e} eV/A^2")
print(f"hc (profundidad de contacto) = {hc:.4f} A")
print(f"H = {H_GPa:.3f} GPa")
