"""
compute_CN_indentation.py
Cambios en el numero de coordinacion (CN) por densificacion bajo la punta,
en las 4 simulaciones de nanoindentacion: {CaO, noCa} x {IEX, sin IEX}.

Compara CN cation-O en dos frames de indent.lammpstrj:
  step 20000  : fin de equilibracion, antes de cargar la punta ("pristino")
  step 220000 : fin del hold a carga maxima ("bajo la punta")

Cutoffs (mismos que COMPARISON/BO_NBO):
  Si-O = 2.00 A   Ca-O = 3.20 A   Na-O = 3.21 A   K-O = 3.77 A

Atom types: Si=1, O=2, Ca=3, Na=4, K=5, C(punta)=6

Genera: CN_indentation.png / .pdf en COMPARISON/CN_INDENTATION/
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import cKDTree

HERE = Path(__file__).parent
IND  = HERE.parent / "INDENTATION"

STEPS = [20000, 220000]
STEP_LABEL = {20000: "pristino", 220000: "bajo la punta"}

TYPE_SI, TYPE_O, TYPE_CA, TYPE_NA, TYPE_K = 1, 2, 3, 4, 5
TYPE_NAME = {TYPE_SI: "Si", TYPE_CA: "Ca", TYPE_NA: "Na", TYPE_K: "K"}
RCUT = {TYPE_SI: 2.00, TYPE_CA: 3.20, TYPE_NA: 3.21, TYPE_K: 3.77}

CASES = [
    ("CaO",  "IEX",     IND / "cao_iex_xt15_r1_300K" / "indent.lammpstrj"),
    ("CaO",  "sin IEX", IND / "cao_x0_r1"            / "indent.lammpstrj"),
    ("noCa", "IEX",     IND / "noca_iex_xt25_r1_300K" / "indent.lammpstrj"),
    ("noCa", "sin IEX", IND / "noca_x0_r1"            / "indent.lammpstrj"),
]

CACHE_DIR = HERE / "frames_cache"
CACHE_DIR.mkdir(exist_ok=True)


def extract_frames(traj_path, steps, cache_tag):
    """Extrae frames completos (id,type,x,y,z + box) de un dump LAMMPS grande,
    en una sola pasada. Cachea en .npz para no re-parsear cada vez."""
    cache_file = CACHE_DIR / f"{cache_tag}.npz"
    if cache_file.exists():
        d = np.load(cache_file, allow_pickle=True)
        return {int(s): (d[f"box_{s}"], d[f"types_{s}"], d[f"pos_{s}"]) for s in steps
                if f"box_{s}" in d}

    targets = set(steps)
    found = {}
    state = "seek_ts"
    cur_step = None
    natoms = None
    box = np.zeros(3)
    lo = np.zeros(3)
    box_idx = 0
    rows = []
    atoms_read = 0

    with open(traj_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "ITEM: TIMESTEP":
                state = "read_ts"
                continue
            if state == "read_ts":
                cur_step = int(line)
                state = "seek_natoms"
                continue
            if line == "ITEM: NUMBER OF ATOMS":
                state = "read_natoms"
                continue
            if state == "read_natoms":
                natoms = int(line)
                state = "seek_box"
                box_idx = 0
                continue
            if line.startswith("ITEM: BOX BOUNDS"):
                state = "read_box"
                box_idx = 0
                continue
            if state == "read_box":
                blo, bhi = map(float, line.split()[:2])
                lo[box_idx] = blo
                box[box_idx] = bhi - blo
                box_idx += 1
                if box_idx == 3:
                    state = "seek_atoms_hdr"
                continue
            if line.startswith("ITEM: ATOMS"):
                atoms_read = 0
                if cur_step in targets:
                    state = "read_atoms"
                    rows = []
                else:
                    state = "skip_atoms"
                continue
            if state == "read_atoms":
                p = line.split()
                rows.append((int(p[1]), float(p[2]), float(p[3]), float(p[4])))
                atoms_read += 1
                if atoms_read == natoms:
                    arr = np.array(rows)
                    pos_wrapped = np.mod(arr[:, 1:4] - lo, box)
                    found[cur_step] = (box.copy(), arr[:, 0].astype(int), pos_wrapped)
                    state = "seek_ts"
                    if len(found) == len(targets):
                        break
                continue
            if state == "skip_atoms":
                atoms_read += 1
                if atoms_read == natoms:
                    state = "seek_ts"
                continue

    # guardar cache
    save_kw = {}
    for s, (b, t, p) in found.items():
        save_kw[f"box_{s}"] = b
        save_kw[f"types_{s}"] = t
        save_kw[f"pos_{s}"] = p
    np.savez_compressed(cache_file, **save_kw)
    return found


def compute_cn(pos, types, box, species_present):
    """CN promedio (cation-O) para cada especie presente."""
    o_pos = pos[types == TYPE_O]
    result = {}
    for sp in species_present:
        cpos = pos[types == sp]
        if len(cpos) == 0 or len(o_pos) == 0:
            continue
        tree = cKDTree(o_pos, boxsize=box)
        counts = tree.query_ball_point(cpos, RCUT[sp], return_length=True)
        result[sp] = np.array(counts, dtype=float)
    return result


def main():
    all_results = {}  # (glass,iex) -> step -> species -> CN array

    for glass, iex, traj in CASES:
        print(f"=== {glass} {iex} ===")
        if not traj.exists():
            print(f"  FALTA: {traj}")
            continue
        tag = f"{glass}_{iex}".replace(" ", "")
        frames = extract_frames(traj, STEPS, tag)

        species_present = [TYPE_SI, TYPE_NA, TYPE_K] + ([TYPE_CA] if glass == "CaO" else [])

        per_step = {}
        for step in STEPS:
            if step not in frames:
                print(f"  frame {step} no encontrado")
                continue
            box, types, pos = frames[step]
            cn = compute_cn(pos, types, box, species_present)
            per_step[step] = cn
            for sp in species_present:
                if sp in cn:
                    print(f"  step={step} {TYPE_NAME[sp]}: <CN>={cn[sp].mean():.3f} (n={len(cn[sp])})")
        all_results[(glass, iex)] = per_step

    # ── Figura: 2x2 paneles (filas=CaO/noCa, cols=IEX/sin IEX), barras CN antes/despues ──
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharey=False)
    row_glass = ["CaO", "noCa"]
    col_iex = ["sin IEX", "IEX"]

    colors = {20000: "#7a9cc6", 220000: "#c0392b"}

    for i, glass in enumerate(row_glass):
        for j, iex in enumerate(col_iex):
            ax = axes[i, j]
            per_step = all_results.get((glass, iex), {})
            species_present = [TYPE_SI, TYPE_NA, TYPE_K] + ([TYPE_CA] if glass == "CaO" else [])
            labels = [TYPE_NAME[sp] for sp in species_present]
            x = np.arange(len(species_present))
            width = 0.35

            for k, step in enumerate(STEPS):
                means = []
                stds = []
                for sp in species_present:
                    cn = per_step.get(step, {}).get(sp)
                    if cn is None:
                        means.append(0); stds.append(0)
                    else:
                        means.append(cn.mean()); stds.append(cn.std())
                offset = (k - 0.5) * width
                ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                       label=STEP_LABEL[step], color=colors[step], zorder=3)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9.5)
            ax.set_title(f"{glass} — {iex}", fontsize=10.5)
            ax.grid(axis='y', lw=0.35, color='#dddddd', zorder=0)
            ax.tick_params(labelsize=9.5)
            if j == 0:
                ax.set_ylabel("CN promedio (cation-O)", fontsize=9.5)
            if i == 0 and j == 0:
                ax.legend(fontsize=8.5, frameon=True, framealpha=0.9, edgecolor='#cccccc')

    fig.tight_layout()
    fig.savefig(HERE / "CN_indentation.png", dpi=200)
    fig.savefig(HERE / "CN_indentation.pdf")
    print(f"\nFiguras guardadas en {HERE}")


if __name__ == "__main__":
    main()
