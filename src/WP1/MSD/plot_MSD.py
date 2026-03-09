#!/usr/bin/env python3
"""
Analyze LAMMPS MSD outputs:
- Reads msd_Na_*K.dat and msd_K_*K.dat
- Plots MSD(t) for each temperature
- Asks the user for a time range (ns) to fit the linear regime
- Computes diffusion coefficients D(T)
- Plots Arrhenius (ln D vs 1/T) and extracts activation energy for Na and K

Assumptions (from your LAMMPS input):
- timestep dt = 0.002 ps (2 fs)  -> change with --dt if needed
- MSD is in Å^2 (LAMMPS compute msd)
- Time column in the file is "step" (timestep index). Script converts step -> time.

Units:
- D is computed from 3D Einstein relation: MSD = 6 D t
- D is output in m^2/s (also Å^2/ps for reference)
- Ea is output in eV and kJ/mol
"""

import argparse
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


K_B_EV_PER_K = 8.617333262e-5  # eV/K
EV_TO_KJ_PER_MOL = 96.4853321233  # kJ/mol per eV


@dataclass
class MSDSeries:
    T: float
    time_ns: np.ndarray
    msd_tot_A2: np.ndarray


def parse_temperature_from_filename(path: str) -> float:
    # Expected like: msd_Na_800K.dat or msd_K_1600K.dat
    m = re.search(r"_([0-9]+(?:\.[0-9]+)?)K", os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse temperature from filename: {path}")
    return float(m.group(1))


def read_msd_file(path: str, dt_ps: float) -> MSDSeries:
    """
    Reads a LAMMPS fix ave/time output file produced with:
      fix ... ave/time ... file msd_*.dat mode vector
    Typical columns (after comments):
      step  msd_x  msd_y  msd_z  msd_tot
    """
    T = parse_temperature_from_filename(path)

    steps = []
    msd_tot = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            # Be robust: allow extra columns; require at least 2 numbers (step and something)
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                continue

            if len(vals) < 2:
                continue

            step = vals[0]
            # Prefer last column as msd_tot if present, else second column
            msd = vals[-1] if len(vals) >= 5 else vals[1]

            steps.append(step)
            msd_tot.append(msd)

    if len(steps) < 5:
        raise RuntimeError(f"Not enough numeric data read from {path}")

    steps = np.array(steps, dtype=float)
    msd_tot = np.array(msd_tot, dtype=float)

    # step -> time
    time_ps = steps * dt_ps
    time_ns = time_ps / 1000.0

    return MSDSeries(T=T, time_ns=time_ns, msd_tot_A2=msd_tot)


def linear_fit_D(time_ns: np.ndarray, msd_A2: np.ndarray, fit_range_ns: Tuple[float, float]) -> Tuple[float, float, float]:
    """
    Fit MSD(t) = a + b t in the chosen range, return:
      slope b in Å^2/ns,
      D in Å^2/ps,
      D in m^2/s
    Using 3D Einstein relation: MSD = 6 D t
    """
    t0, t1 = fit_range_ns
    if t1 <= t0:
        raise ValueError("fit_range_ns must have end > start")

    mask = (time_ns >= t0) & (time_ns <= t1)
    if mask.sum() < 5:
        raise RuntimeError("Not enough points in the chosen fit range. Choose a wider range.")

    t = time_ns[mask]
    y = msd_A2[mask]

    # Linear regression via polyfit
    b_A2_per_ns, a_A2 = np.polyfit(t, y, 1)

    # Convert slope to Å^2/ps
    b_A2_per_ps = b_A2_per_ns / 1000.0

    # D in Å^2/ps
    D_A2_per_ps = b_A2_per_ps / 6.0

    # 1 Å^2/ps = 1e-8 m^2/s
    D_m2_per_s = D_A2_per_ps * 1e-8

    return b_A2_per_ns, D_A2_per_ps, D_m2_per_s


def r2_score(x: np.ndarray, y: np.ndarray, a: float, b: float) -> float:
    # yhat = a + b x
    yhat = a + b * x
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def plot_msd(series_by_T: Dict[float, MSDSeries], species: str, outdir: str) -> str:
    Ts = sorted(series_by_T.keys())
    plt.figure()
    for T in Ts:
        s = series_by_T[T]
        plt.plot(s.time_ns, s.msd_tot_A2, label=f"{int(T)} K")
    plt.xlabel("Time (ns)")
    plt.ylabel(r"MSD$_{tot}$ ($\AA^2$)")
    plt.title(f"MSD vs time ({species})")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"MSD_{species}.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath


def arrhenius_fit(Ts: np.ndarray, Ds_m2_s: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Fit ln(D) = ln(D0) - Ea/(kB*T)
    Returns:
      Ea_eV, Ea_kJmol, lnD0, r2
    """
    x = 1.0 / Ts
    y = np.log(Ds_m2_s)

    m, c = np.polyfit(x, y, 1)  # y = m x + c, here m = -Ea/kB
    Ea_eV = -m * K_B_EV_PER_K
    Ea_kJmol = Ea_eV * EV_TO_KJ_PER_MOL

    # r2
    yhat = m * x + c
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return Ea_eV, Ea_kJmol, c, r2


def plot_arrhenius(Ts: np.ndarray, Ds_m2_s: np.ndarray, species: str, outdir: str) -> str:
    x = 1.0 / Ts
    y = np.log(Ds_m2_s)

    m, c = np.polyfit(x, y, 1)
    xfit = np.linspace(x.min(), x.max(), 200)
    yfit = m * xfit + c

    plt.figure()
    plt.plot(x, y, "o", label="Data")
    plt.plot(xfit, yfit, "-", label="Linear fit")
    plt.xlabel(r"1/T (K$^{-1}$)")
    plt.ylabel(r"ln(D)  [D in m$^2$/s]")
    plt.title(f"Arrhenius plot ({species})")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"Arrhenius_{species}.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath


def find_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    return [f for f in files if os.path.isfile(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.002, help="Timestep in ps (default: 0.002 ps)")
    ap.add_argument("--indir", type=str, default=".", help="Directory containing msd_*.dat files")
    ap.add_argument("--outdir", type=str, default="msd_analysis", help="Output directory for plots/results")
    ap.add_argument("--na_glob", type=str, default="msd_Na_*K.dat", help="Glob for Na MSD files")
    ap.add_argument("--k_glob", type=str, default="msd_K_*K.dat", help="Glob for K MSD files")
    args = ap.parse_args()

    indir = args.indir
    outdir = args.outdir

    na_files = find_files(os.path.join(indir, args.na_glob))
    k_files = find_files(os.path.join(indir, args.k_glob))

    if not na_files and not k_files:
        raise SystemExit("No MSD files found. Check --indir and globs (e.g., msd_Na_800K.dat).")

    def load_species(files: List[str], label: str) -> Dict[float, MSDSeries]:
        by_T = {}
        for f in files:
            s = read_msd_file(f, dt_ps=args.dt)
            by_T[s.T] = s
        if by_T:
            print(f"Loaded {len(by_T)} temperature points for {label}: {sorted(by_T.keys())}")
        return by_T

    na_byT = load_species(na_files, "Na")
    k_byT = load_species(k_files, "K")

    # Plot MSD curves
    if na_byT:
        p = plot_msd(na_byT, "Na", outdir)
        print(f"Saved: {p}")
    if k_byT:
        p = plot_msd(k_byT, "K", outdir)
        print(f"Saved: {p}")

    # Ask fit range(s)
    def ask_range(species: str) -> Tuple[float, float]:
        print(f"\nChoose fit range for {species} to compute D from MSD(t)")
        print("Enter start and end time in ns (e.g., 2 10).")
        while True:
            try:
                raw = input(f"{species} fit range [ns] (start end): ").strip()
                a, b = raw.split()
                t0, t1 = float(a), float(b)
                if t1 <= t0:
                    print("End must be > start. Try again.")
                    continue
                return (t0, t1)
            except Exception:
                print("Could not parse. Please type two numbers, e.g. 2 10.")

    results = {}

    for species, byT in [("Na", na_byT), ("K", k_byT)]:
        if not byT:
            continue

        fit_range = ask_range(species)

        Ts = []
        Ds_A2_ps = []
        Ds_m2_s = []

        # Also save per-T fit diagnostics
        perT_rows = []

        for T in sorted(byT.keys()):
            s = byT[T]
            # Fit
            t0, t1 = fit_range
            mask = (s.time_ns >= t0) & (s.time_ns <= t1)
            t = s.time_ns[mask]
            y = s.msd_tot_A2[mask]
            if mask.sum() < 5:
                print(f"[WARN] {species} {T}K: not enough points in range {fit_range}; skipping.")
                continue

            b_A2_per_ns, D_A2_ps, D_m2_s = linear_fit_D(s.time_ns, s.msd_tot_A2, fit_range)
            # R2 of fit
            b, a = np.polyfit(t, y, 1)
            r2 = r2_score(t, y, a=a, b=b)

            Ts.append(T)
            Ds_A2_ps.append(D_A2_ps)
            Ds_m2_s.append(D_m2_s)

            perT_rows.append((T, b_A2_per_ns, D_A2_ps, D_m2_s, r2))

        Ts = np.array(Ts, dtype=float)
        Ds_m2_s = np.array(Ds_m2_s, dtype=float)

        if len(Ts) < 2:
            print(f"[WARN] Not enough temperatures to do Arrhenius fit for {species}.")
            continue

        # Arrhenius
        Ea_eV, Ea_kJmol, lnD0, r2A = arrhenius_fit(Ts, Ds_m2_s)
        arr_plot = plot_arrhenius(Ts, Ds_m2_s, species, outdir)

        # Save table
        os.makedirs(outdir, exist_ok=True)
        table_path = os.path.join(outdir, f"D_vs_T_{species}.csv")
        with open(table_path, "w", encoding="utf-8") as w:
            w.write("T_K,slope_A2_per_ns,D_A2_per_ps,D_m2_per_s,R2_linear\n")
            for row in perT_rows:
                w.write(f"{row[0]:.0f},{row[1]:.8e},{row[2]:.8e},{row[3]:.8e},{row[4]:.6f}\n")

        results[species] = (Ea_eV, Ea_kJmol, lnD0, r2A, arr_plot, table_path)

    # Print summary
    print("\n==================== SUMMARY ====================")
    for species in ["Na", "K"]:
        if species not in results:
            continue
        Ea_eV, Ea_kJmol, lnD0, r2A, arr_plot, table_path = results[species]
        print(f"{species}:")
        print(f"  Activation energy Ea = {Ea_eV:.4f} eV  = {Ea_kJmol:.2f} kJ/mol")
        print(f"  Arrhenius fit R^2    = {r2A:.5f}")
        print(f"  Saved Arrhenius plot = {arr_plot}")
        print(f"  Saved D(T) table     = {table_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
