import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ================== Settings ==================
filename = "tip_force_vs_disp.dat"
apex_angle_deg = 77.32      # total included cone angle φ
epsilon_geom = 0.72         # ε for conical indenter (Oliver–Pharr)
fit_first_frac = 0.35       # fraction of unloading near Pmax used for fit
min_fit_points = 20         # minimum points for fit
dh_tol_A = 0.01             # Å threshold to detect end of HOLD and start of true unloading
area_coeff = 'pi'           # 'pi' => Ac = π hc^2 tan^2(φ/2); 'user' => 4 hc^2 tan^2(φ/2)
# ===================================================

EV_PER_ANG_TO_N = 1.602176634e-9
ANG_TO_M = 1e-10
PA_TO_GPA = 1e-9

def P_model(h, a, m, hf_fixed):
    # P = a * (h - hf)^m with hf fixed by user
    return a * np.maximum(h - hf_fixed, 0.0)**m

# -------- Load data --------
data = np.loadtxt(filename, comments="#")
t  = data[:,0]
F0 = data[:,1]  # eV/Å (units metal)
z  = data[:,2]  # Å

# Make loading positive (flip if compressive is negative)
tail = max(5, len(F0)//10)
F = -F0 if np.median(F0[-tail:]) < 0 else F0

# -------- Detect contact --------
n0 = max(10, int(0.10*len(F)))
noise = np.median(np.abs(F[:n0])) if n0 > 0 else 0.0
Fmax_abs = np.max(np.abs(F)) if len(F) else 0.0
thr = max(5.0*noise, 0.01*Fmax_abs) if Fmax_abs > 0 else 0.0
i_contact = np.argmax(np.abs(F) >= thr)
if not np.any(np.abs(F) >= thr):
    raise RuntimeError("Contact not detected.")
z_contact = z[i_contact]
h = z_contact - z  # indentation depth (Å), zero at contact

# -------- Pmax, hmax, start of unloading (skip HOLD plateau) --------
i_pmax = int(np.argmax(F))
Pmax = float(F[i_pmax])
hmax = float(h[i_pmax])

start_unload = i_pmax
for k in range(i_pmax + 1, len(h)):
    if (h[i_pmax] - h[k]) > dh_tol_A:  # retraction starts
        start_unload = k
        break

h_un = h[start_unload:]
P_un = F[start_unload:]
if len(h_un) < min_fit_points:
    raise RuntimeError("Unloading segment too short after HOLD.")

# -------- Plot to let user pick hf (no automatic guess) --------
plt.figure(figsize=(6,4))
plt.plot(h, F, 'o', ms=3, alpha=0.5, label='All data')
plt.plot(h_un, P_un, 'o', ms=3, alpha=0.9, label='Unloading')
plt.axvline(hmax, color='k', ls=':', lw=1.0, label='hmax')
plt.xlabel("Indentation depth h (Å)")
plt.ylabel("Load P (eV/Å)")
plt.title("Inspect unloading, then close window and enter hf (Å)")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# -------- Ask user for hf --------
while True:
    s = input("Enter hf (residual depth) in Å: ").strip()
    try:
        hf = float(s)
        break
    except ValueError:
        print("Invalid value. Please enter a number (e.g., 25.4).")

# -------- Fit FIRST fraction of unloading with hf fixed --------
n_first = max(min_fit_points, int(fit_first_frac * len(h_un)))
h_fit = h_un[:n_first]
P_fit = P_un[:n_first]

m0 = 1.5
span = max(1e-6, np.max(h_fit) - hf)
a0 = max(np.max(P_fit), 1e-6) / (span**m0)
p0 = [a0, m0]
bounds = ([0.0, 0.9], [np.inf, 3.0])

def fit_func(hvals, a, m):
    return P_model(hvals, a, m, hf)

popt, pcov = curve_fit(fit_func, h_fit, P_fit, p0=p0, bounds=bounds, maxfev=20000)
a_fit, m_fit = [float(x) for x in popt]

# -------- Stiffness at hmax, hc, Ac, Hardness --------
arg = max(hmax - hf, 0.0)
S_eV_per_A2 = a_fit * m_fit * (arg**(m_fit - 1.0)) if arg > 0 else np.nan

hc = hmax - epsilon_geom * Pmax / S_eV_per_A2
theta = np.deg2rad(apex_angle_deg / 2.0)
if area_coeff == 'pi':
    Ac = np.pi * (hc * ANG_TO_M)**2 * (np.tan(theta)**2)
else:  # 'user'
    Ac = 4.0 * (hc * ANG_TO_M)**2 * (np.tan(theta)**2)

Pmax_N = Pmax * EV_PER_ANG_TO_N
H_GPa  = (Pmax_N / Ac) * PA_TO_GPA

# -------- Results --------
print("\n=== Oliver–Pharr (hf provided by user; fit on FIRST unloading fraction) ===")
print(f"Pmax (eV/Å)          : {Pmax:.6e}")
print(f"hmax (Å)             : {hmax:.6f}")
print(f"hf  (Å, user)        : {hf:.6f}")
print(f"Fit (a, m)           : a={a_fit:.6e}, m={m_fit:.4f}")
print(f"S (eV/Å^2)           : {S_eV_per_A2:.6e}")
print(f"hc (Å)               : {hc:.6f}")
print(f"Ac formula           : {'π' if area_coeff=='pi' else '4'} * hc^2 * tan^2(φ/2)  (φ={apex_angle_deg:.2f}°)")
print(f"H (GPa)              : {H_GPa:.3f}")

# -------- Final diagnostic plot --------
plt.figure(figsize=(6,4))
plt.plot(h, F, 'o', ms=3, alpha=0.5, label='All data')
plt.plot(h_un, P_un, 'o', ms=3, alpha=0.9, label='Unloading')
plt.plot(h_fit, P_fit, 'o', ms=4, label='Fit (first fraction)')
h_line = np.linspace(min(h_fit), max(h_fit), 200)
plt.plot(h_line, P_model(h_line, a_fit, m_fit, hf), '-', lw=2, label='Fit (hf fixed)')
plt.axvline(hf,   color='crimson', ls='--', lw=1.5, label=f'hf = {hf:.3f} Å')
plt.axvline(hmax, color='k',       ls=':',  lw=1.0, label='hmax')
plt.xlabel("Indentation depth h (Å)")
plt.ylabel("Load P (eV/Å)")
plt.title("Unloading fit with user-provided hf")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()
