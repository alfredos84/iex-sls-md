# compute_rdf_multi_final.py
# OVITO 3.14 - robusto para pares de cualquier orden

from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier
import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    "font.family": "serif",  # Computer Modern Roman
    "mathtext.fontset": "cm",  # Usa símbolos LaTeX-like
    "font.size": 15
})

plt.figure(figsize=(6,6))

# ========== CONFIG ==========
basepath = "../MELTQUENCH/"
traj_pattern = "Quenched_glass_Tfinal_300_x{}.lammpstrj"
x_values = [0, 1, 2, 3, 4, 5, 6, 9, 12, 15]
cutoff = 6.0
nbins = 200
Nframes_avg = 200
figsize = (6, 6)
atom_map = {"1": "Si", "2": "O", "3": "Ca", "4": "Na", "5": "K"}
pairs = ["1-2", "2-2", "3-2", "4-2", "5-2", "1-4", "1-5"]

outdir = "rdf_results"
os.makedirs(outdir, exist_ok=True)

# ========== DETECTA TIPOS POR x ==========
tipos_por_x = {}
for x in x_values:
    traj_file = os.path.join(basepath, traj_pattern.format(x))
    pipeline = import_file(traj_file, multiple_frames=True)
    data = pipeline.compute(0)
    types = np.unique(data.particles['Particle Type'])
    tipos_por_x[x] = [atom_map[str(i)] for i in types if str(i) in atom_map]
    print(f"x{x}: tipos presentes = {tipos_por_x[x]}")
print("\n========================\n")

# ========== FUNCIONES ==========
def get_pair_col(col_names, a, b):
    p1 = f"{a}-{b}"
    p2 = f"{b}-{a}"
    if p1 in col_names:
        return col_names.index(p1), p1
    elif p2 in col_names:
        return col_names.index(p2), p2
    else:
        return None, None

def compute_avg_rdf(traj_file, a, b):
    pipeline = import_file(traj_file, multiple_frames=True)
    pipeline.modifiers.append(
        CoordinationAnalysisModifier(
            cutoff=cutoff,
            number_of_bins=nbins,
            partial=True
        )
    )
    nframes = pipeline.source.num_frames
    start = max(0, nframes - Nframes_avg)
    r_vals, sum_vals, count = None, None, 0

    for frame in range(start, nframes):
        data = pipeline.compute(frame)
        arr = np.asarray(data.tables['coordination-rdf'].xy())
        r = arr[:, 0]
        if r_vals is None:
            r_vals = r
        col_names = ["r"] + list(data.tables['coordination-rdf'].y.component_names)
        idx, found_pair = get_pair_col(col_names, a, b)
        if idx is None:
            return None, None, None
        gvals = arr[:, idx]
        if sum_vals is None:
            sum_vals = np.zeros_like(gvals)
        sum_vals += gvals
        count += 1

    return r_vals, sum_vals / count, found_pair

# ========== LOOP PRINCIPAL ==========
for pair in pairs:
    a, b = pair.split("-")
    elem_a, elem_b = atom_map.get(a, a), atom_map.get(b, b)
    plt.figure(figsize=figsize)
    plotted = False

    for x in x_values:
        traj_file = os.path.join(basepath, traj_pattern.format(x))
        presentes = tipos_por_x[x]
        r_vals, avg_vals, found_pair = compute_avg_rdf(traj_file, a, b)
        if r_vals is None:
            print(f"[INFO] Par {elem_a}-{elem_b} ({a}-{b}/{b}-{a}) no existe en x{x} ({presentes}), se omite.")
            continue

        outname = f"{elem_a}_{elem_b}_x{x}.dat"
        outpath = os.path.join(outdir, outname)
        np.savetxt(outpath, np.column_stack([r_vals, avg_vals]),
                   header=f"r [Å]   g({elem_a}-{elem_b}) [{found_pair}]", comments="")
        print(f"[OK] Guardado: {outpath}")

        leyenda = f"x={int(x)}"
        plt.plot(r_vals, avg_vals, label=leyenda, linewidth=1.0)
        plotted = True

    if plotted:
        plt.xlabel("r [Å]")
        plt.ylabel(f"g(r) {elem_a}-{elem_b}")
        plt.title(f"g(r) {elem_a}-{elem_b}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        fig_name = os.path.join(outdir, f"{elem_a}_{elem_b}_rdf.png")
        fig_png  = fig_name + ".png"
        fig_pdf  = fig_name + ".pdf"
        plt.savefig(fig_png, dpi=300)
        plt.savefig(fig_pdf)
        plt.close()
        print(f"[OK] Figura guardada: {fig_name}")
    else:
        print(f"[WARN] Ningún x tenía el par {elem_a}-{elem_b}, no se genera figura.")

# ========== RESUMEN FINAL ==========
print("\n========================\nRESUMEN DE TIPOS PRESENTES EN CADA x:")
for x in x_values:
    print(f"x{x}: {tipos_por_x[x]}")
