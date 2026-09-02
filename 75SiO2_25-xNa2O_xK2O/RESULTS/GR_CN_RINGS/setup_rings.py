"""
setup_rings.py
Prepara directorios, archivos TRJ e input files para RINGS, y corre RINGS.

Para cada (protocolo, x):
  - Lee 3 réplicas de LAMMPS .data  (NS=3, modo legado)
  - O bien lee trayectorias NVT+NPT dump (NS=n_frames×3, modo traj)
  - Combina snapshots en un archivo TRJ y corre RINGS

Protocolos con trayectoria dump:
  IEX2   : IOX2_{NVT,NPT}_723K_xp{x}_r{r}.lammpstrj   (STAGE4)
  REIEX2 : REIEX2_{NVT,NPT}_723K_xp{x}_r{r}.lammpstrj (STAGE12)

Dump column order: id type q x y z
"""

import numpy as np
import subprocess
import os
from pathlib import Path

HERE    = Path(__file__).parent
BASE    = HERE.parent.parent
RINGS_BIN = BASE / "rings-code-v1.3.5" / "src" / "rings"

AM_DIR     = BASE / "STAGE1_MELTQUENCH"     / "data" / "asmelted_723K"
IEX1_DIR   = BASE / "STAGE3_IEX_PROTO1"    / "data" / "iox1_723K"
IEX2_DIR   = BASE / "STAGE4_IEX_PROTO2"    / "data" / "iox2_723K"
IEX2_TRJ   = BASE / "STAGE4_IEX_PROTO2"    / "trajectories"
REIEX1_DIR = BASE / "STAGE11_REVERSE_IEX1" / "data" / "rev_iex1_723K"
REIEX2_DIR = BASE / "STAGE12_REVERSE_IEX2" / "data" / "rev_iex2_723K"
REIEX2_TRJ = BASE / "STAGE12_REVERSE_IEX2" / "trajectories"

PROTOCOLS = {
    'AM':     {'dir': AM_DIR,     'x_list': [0,1,3,6,9,12,15],
               'fname': lambda x,r: f"AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data"},
    'IEX1':   {'dir': IEX1_DIR,   'x_list': [1,3,6,9,12,15],
               'fname': lambda x,r: f"IOX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data"},
    'IEX2':   {'dir': IEX2_DIR,   'x_list': [0,1,3,6,9,12],
               'fname': lambda x,r: f"IOX2_723K_xp{x}_r{r}_PMMCS_rc8p0.data",
               'traj_dir': IEX2_TRJ,
               'traj_nvt': lambda x,r: f"IOX2_NVT_723K_xp{x}_r{r}_PMMCS_rc8p0.lammpstrj",
               'traj_npt': lambda x,r: f"IOX2_NPT_723K_xp{x}_r{r}_PMMCS_rc8p0.lammpstrj"},
    'REIEX1': {'dir': REIEX1_DIR, 'x_list': [1,3,6,9,12,15],
               'fname': lambda x,r: f"REIEX1_723K_xt{x}_r{r}_PMMCS_rc8p0.data"},
    'REIEX2': {'dir': REIEX2_DIR, 'x_list': [0,1,3,6,9,12],
               'fname': lambda x,r: f"REIEX2_723K_xp{x}_r{r}_PMMCS_rc8p0.data",
               'traj_dir': REIEX2_TRJ,
               'traj_nvt': lambda x,r: f"REIEX2_NVT_723K_xp{x}_r{r}_PMMCS_rc8p0.lammpstrj",
               'traj_npt': lambda x,r: f"REIEX2_NPT_723K_xp{x}_r{r}_PMMCS_rc8p0.lammpstrj"},
}

# Atom types → RINGS species labels
# LAMMPS: Si=1, O=2, Ca=3, Na=4, K=5
TYPE_TO_SP = {1:'Si', 2:'O ', 3:'Ca', 4:'Na', 5:'K '}
SP_ORDER   = [1, 2, 3, 4, 5]   # order in TRJ file

# g(r) parameters
DR          = 0.02    # Å — resolución deseada
R_MAX       = 8.0     # Å

ANG_TO_BOHR = 1.88972612   # 1 Å en bohr (RINGS espera coordenadas en unidades atómicas)

# Ring parameters
TAILLE_REF  = 20     # max ring size in Si atoms — hasta 6-Si (12 átomos), suficiente en SLS
TAILLC      = 3

# Cutoffs para g(r) y CN (todos los pares relevantes)
CUTOFFS_GR = {
    ('Si','Si'): 3.2,
    ('Si','O '): 2.1,
    ('Si','Ca'): 0.01,
    ('Si','Na'): 0.01,
    ('Si','K '): 0.01,
    ('O ','O '): 3.2,
    ('O ','Ca'): 3.5,
    ('O ','Na'): 3.5,
    ('O ','K '): 4.0,
    ('Ca','Ca'): 0.01,
    ('Ca','Na'): 0.01,
    ('Ca','K '): 0.01,
    ('Na','Na'): 4.5,
    ('Na','K '): 4.5,
    ('K ','K '): 5.0,
}

# Cutoffs para anillos: solo Si-O (ABAB alternating, solo se sigue Si-O)
# Todos los demás pares en 0.01 para evitar el prompt interactivo pero sin crear conexiones
CUTOFFS_RING = {
    ('Si','Si'): 0.01,
    ('Si','O '): 2.1,   # único enlace que importa para Si-O-Si-O rings
    ('Si','Ca'): 0.01,
    ('Si','Na'): 0.01,
    ('Si','K '): 0.01,
    ('O ','O '): 0.01,
    ('O ','Ca'): 0.01,
    ('O ','Na'): 0.01,
    ('O ','K '): 0.01,
    ('Ca','Ca'): 0.01,
    ('Ca','Na'): 0.01,
    ('Ca','K '): 0.01,
    ('Na','Na'): 0.01,
    ('Na','K '): 0.01,
    ('K ','K '): 0.01,
}

CUTOFFS = CUTOFFS_GR   # default (se sobreescribe por argumento)


# ── Leer trayectoria LAMMPS dump ─────────────────────────────────────────────
# Columnas esperadas: id type q x y z  (verificado en STAGE4 / STAGE12)

def read_lammps_dump(path):
    """
    Lee todos los frames de un archivo dump LAMMPS.
    Retorna lista de (box_array[3], atoms_array[N,4]) donde atoms[:,0]=type, [:,1:4]=xyz.
    """
    frames = []
    box    = np.zeros(3)
    atoms  = []
    n_atoms = 0
    state  = 'header'
    col_type = col_x = col_y = col_z = None

    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if line == 'ITEM: TIMESTEP':
                if atoms:
                    frames.append((box.copy(), np.array(atoms, dtype=float)))
                box   = np.zeros(3)
                atoms = []
                state = 'skip_ts'
            elif state == 'skip_ts':
                state = 'expect_natoms'
            elif line == 'ITEM: NUMBER OF ATOMS':
                state = 'read_natoms'
            elif state == 'read_natoms':
                n_atoms = int(line)
                state   = 'expect_box'
            elif line.startswith('ITEM: BOX BOUNDS'):
                state = 'read_box'
                box_idx = 0
            elif state == 'read_box':
                lo, hi = map(float, line.split()[:2])
                box[box_idx] = hi - lo
                box_idx += 1
                if box_idx == 3:
                    state = 'expect_atoms'
            elif line.startswith('ITEM: ATOMS'):
                cols = line.split()[2:]   # labels after 'ITEM: ATOMS'
                col_type = cols.index('type')
                col_x    = cols.index('x')
                col_y    = cols.index('y')
                col_z    = cols.index('z')
                state    = 'read_atoms'
            elif state == 'read_atoms' and line:
                p = line.split()
                atoms.append([float(p[col_type]),
                               float(p[col_x]),
                               float(p[col_y]),
                               float(p[col_z])])
        # último frame
        if atoms:
            frames.append((box.copy(), np.array(atoms, dtype=float)))

    return frames   # list of (box[3], atoms[N,4])


# ── Leer archivo LAMMPS .data ─────────────────────────────────────────────────

def read_lammps_data(path):
    box  = {}
    atoms = []
    in_atoms = False
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            s = line.strip()
            if 'xlo xhi' in s:
                p = s.split(); box['x'] = (float(p[0]), float(p[1]))
            elif 'ylo yhi' in s:
                p = s.split(); box['y'] = (float(p[0]), float(p[1]))
            elif 'zlo zhi' in s:
                p = s.split(); box['z'] = (float(p[0]), float(p[1]))
            elif s.startswith('Atoms'):
                in_atoms = True
            elif in_atoms and s:
                p = s.split()
                if len(p) >= 6:
                    try:
                        atoms.append((int(p[1]),
                                      float(p[3]), float(p[4]), float(p[5])))
                    except ValueError:
                        pass
    Lx = box['x'][1] - box['x'][0]
    Ly = box['y'][1] - box['y'][0]
    Lz = box['z'][1] - box['z'][0]
    return np.array([Lx, Ly, Lz]), np.array(atoms)


# ── Contar átomos por especie ─────────────────────────────────────────────────

def atom_counts(atoms, x_val):
    """Retorna dict {lammps_type: count}, excluyendo tipos con 0 átomos."""
    counts = {}
    for t in SP_ORDER:
        n = int((atoms[:, 0] == t).sum())
        if n > 0:
            counts[t] = n
    return counts


# ── Escribir TRJ ─────────────────────────────────────────────────────────────

def write_trj(trj_path, snapshots, boxes, counts):
    """
    snapshots: list of atom arrays (one per replica/step)
    boxes: list of [Lx,Ly,Lz] (one per step)
    counts: {type: N} — species present
    """
    sp_present = [t for t in SP_ORDER if t in counts]
    with open(trj_path, 'w') as f:
        for atoms, box in zip(snapshots, boxes):
            # CPMD TRJ format: index X Y Z (NOC is INTEGER in RINGS/parameters.F90)
            iatom = 1
            for t in sp_present:
                mask = atoms[:, 0] == t
                pos  = atoms[mask, 1:4]
                for row in pos:
                    # CPMD TRJ: index X Y Z VX VY VZ en unidades atómicas (bohr)
                    # RINGS convierte bohr→Å internamente (ATOMIC_UNITS=.true.)
                    bx, by, bz = row[0]*ANG_TO_BOHR, row[1]*ANG_TO_BOHR, row[2]*ANG_TO_BOHR
                    f.write(f"{iatom:6d}  {bx:16.10f}  {by:16.10f}  {bz:16.10f}"
                            f"  0.0000000000  0.0000000000  0.0000000000\n")
                    iatom += 1


# ── Escribir rings.inp ────────────────────────────────────────────────────────

def write_input(inp_path, trj_name, counts, box, ns, cutoffs=None):
    if cutoffs is None:
        cutoffs = CUTOFFS_GR
    sp_present  = [t for t in SP_ORDER if t in counts]
    sp_labels   = [TYPE_TO_SP[t] for t in sp_present]
    nsp         = len(sp_present)
    na          = sum(counts[t] for t in sp_present)
    Lx, Ly, Lz = box

    # N_DR: número de bins para cubrir L/sqrt(2) con paso Δr=DR
    # g(r) RINGS cubre [0, L/sqrt(2)] en N_DR puntos
    import math
    n_dr = max(80, int(Lx / (DR * math.sqrt(2.0))) + 1)

    # Pares de cutoffs
    pairs = []
    for i in range(nsp):
        for j in range(i, nsp):
            l1, l2 = sp_labels[i], sp_labels[j]
            key = (l1, l2) if (l1, l2) in cutoffs else (l2, l1)
            dc  = cutoffs.get(key, 0.01)
            pairs.append((l1.strip(), l2.strip(), dc))

    with open(inp_path, 'w') as f:
        f.write("# RINGS input — PMMCS SLS glass\n")
        f.write("# Generated by setup_rings.py\n")
        f.write("#\n")
        f.write("SLS-PMMCS\n")
        f.write(f"{na}\n")
        f.write(f"{nsp}\n")
        f.write("  ".join(l.strip() for l in sp_labels) + "\n")
        f.write(f"{ns}\n")
        # LATTYPE=2: a b c on one line, then alpha beta gamma
        f.write("2\n")
        f.write(f"{Lx:.6f}  {Ly:.6f}  {Lz:.6f}\n")
        f.write("90.0  90.0  90.0\n")
        f.write("0.002\n")       # DELTA_T (ps, no afecta g(r))
        f.write("TRJ\n")
        f.write(f"{trj_name}\n")
        for t in sp_present:
            f.write(f"{TYPE_TO_SP[t].strip()}  {counts[t]}\n")
        f.write(f"{n_dr}\n")     # NUMBER_OF_DELTA_R
        f.write("0\n")           # NUMBER_OF_Q_POINTS (no S(q))
        f.write("0.0\n")         # QMAX
        f.write("0.0\n")         # SIGMA_LISS
        f.write("0\n")           # NUMBER_OF_DELTA_ANG
        f.write("0\n")           # NDV
        f.write(f"{TAILLE_REF}\n")
        f.write(f"{TAILLC}\n")
        f.write("\n")            # blank line before cutoffs
        for l1, l2, dc in pairs:
            f.write(f"{l1}  {l2}  {dc:.4f}\n")
        f.write(f"To  {R_MAX:.2f}\n")   # ATOT + Gr_cutoff


# ── Escribir options ──────────────────────────────────────────────────────────

def write_options(opt_path, do_rings=True):
    rings_str = '.true. ' if do_rings else '.false.'
    with open(opt_path, 'w') as f:
        f.write("# RINGS options file\n")         # blank read 1
        f.write("# Generated by setup_rings.py\n") # blank read 2
        f.write("#\n")                             # blank read 3
        f.write("PBC          .true.\n")
        f.write("FRAC         .false.\n")
        f.write("CALC_GR      .true.\n")
        f.write("CALC_SQ      .false.\n")
        f.write("CALC_SK      .false.\n")
        f.write("CALC_GK      .false.\n")
        f.write("CALC_MSD     .false.\n")
        f.write("MSDEA        .false.\n")
        f.write("CALC_BONDS   .true.\n")
        f.write("CALC_ANG     .false.\n")
        f.write("CALC_CHAINS  .false.\n")
        f.write("#\n")                             # blank separator
        f.write("CTLT         0\n")
        f.write("AAAA         .false.\n")
        f.write("ACAC         .false.\n")
        f.write("ISOLATED     .false.\n")
        f.write("#\n")                             # blank separator
        f.write(f"CALC_R       {rings_str}\n")
        f.write("#\n")                             # blank separator
        f.write("LTLT         1\n")
        f.write("ABAB         .false.\n")   # no alternado; más rápido; con solo Si-O las cadenas son naturalmente alternantes
        f.write("CALC_R0      .false.\n")
        f.write("CALC_R1      .false.\n")
        f.write(f"CALC_R2      {rings_str}\n")   # King's rings — igual info, significativamente más rápido que primitivos
        f.write("CALC_R3      .false.\n")
        f.write("CALC_R4      .false.\n")
        f.write("CALC_PRINGS  .false.\n")
        f.write("CALC_STRINGS .false.\n")
        f.write("BARYCROUT    .false.\n")
        f.write("RING_P1      .false.\n")
        f.write("RING_P2      .false.\n")
        f.write("RING_P3      .false.\n")
        f.write("RING_P4      .false.\n")
        f.write("RING_P5      .false.\n")
        f.write("#\n")                             # blank separator
        f.write("CALC_VAC     .false.\n")
        f.write("#\n")                             # blank separator (x3)
        f.write("#\n")
        f.write("#\n")
        f.write("EVOLOUT      .false.\n")
        f.write("DXOUT        .false.\n")
        f.write("#\n")                             # blank separator
        f.write("RADOUT       .false.\n")
        f.write("RNGOUT       .false.\n")
        f.write("DRNGOUT      .false.\n")
        f.write("VACOUT       .false.\n")
        f.write("DXTETRA      .false.\n")
        f.write("PATHDX       .false.\n")
        f.write("#\n")                             # blank separator
        f.write("NOM_OUT      output\n")


# ── Setup + run ───────────────────────────────────────────────────────────────

def setup_system(proto, x, do_rings=True):
    info    = PROTOCOLS[proto]
    x_list  = info['x_list']
    if x not in x_list:
        return
    data_dir = info['dir']
    fname_fn = info['fname']

    run_dir = HERE / f"{proto}_x{x}"
    run_dir.mkdir(exist_ok=True)
    (run_dir / "data").mkdir(exist_ok=True)
    (run_dir / "gr").mkdir(exist_ok=True)
    (run_dir / "rstat").mkdir(exist_ok=True)

    data_subdir = run_dir / "data"

    # ── Modo trayectoria: NVT + NPT dump files ────────────────────────────────
    if 'traj_dir' in info:
        traj_dir  = info['traj_dir']
        nvt_fn    = info['traj_nvt']
        npt_fn    = info['traj_npt']

        all_snapshots, all_boxes = [], []
        for r in (1, 2, 3):
            for phase_fn in (npt_fn,):   # solo NPT (equilibrado); NVT omitido
                p = traj_dir / phase_fn(x, r)
                if not p.exists():
                    print(f"  MISSING traj: {p}")
                    continue
                frames = read_lammps_dump(p)
                for box, atoms in frames:
                    # convierte atoms[:,0] a int type; reshape como read_lammps_data
                    at = np.column_stack([atoms[:, 0].astype(int),
                                          atoms[:, 1], atoms[:, 2], atoms[:, 3]])
                    all_snapshots.append(at)
                    all_boxes.append(box)

        if not all_snapshots:
            print(f"  No traj data for {proto} x={x}")
            return

        counts = atom_counts(all_snapshots[0], x)
        box_avg = np.mean(all_boxes, axis=0)
        ns = len(all_snapshots)

        trj_name = f"traj_{ns}frames.trj"
        write_trj(data_subdir / trj_name, all_snapshots, all_boxes, counts)
        write_input(run_dir / "rings_gr.inp", trj_name, counts, box_avg, ns=ns,
                    cutoffs=CUTOFFS_GR)
        write_options(run_dir / "options_gr", do_rings=False)

        print(f"  Setup (traj): {proto} x={x} — {ns} frames, box={box_avg[0]:.3f} Å")
        return run_dir, counts

    # ── Modo legado: 3 archivos .data estáticos ───────────────────────────────
    snapshots, boxes, counts_list = [], [], []
    for r in (1, 2, 3):
        p = data_dir / fname_fn(x, r)
        if not p.exists():
            print(f"  MISSING: {p}")
            continue
        box, atoms = read_lammps_data(p)
        snapshots.append(atoms)
        boxes.append(box)
        counts_list.append(atom_counts(atoms, x))

    if not snapshots:
        print(f"  No data for {proto} x={x}")
        return

    counts  = counts_list[0]
    box_avg = np.mean(boxes, axis=0)

    trj_path = data_subdir / "traj_3rep.trj"
    write_trj(trj_path, snapshots, boxes, counts)

    trj_ring_path = data_subdir / "traj_1rep.trj"
    write_trj(trj_ring_path, snapshots[:1], boxes[:1], counts)

    write_input(run_dir / "rings_gr.inp",   "traj_3rep.trj", counts, box_avg, ns=3,
                cutoffs=CUTOFFS_GR)
    write_options(run_dir / "options_gr",   do_rings=False)

    if do_rings:
        write_input(run_dir / "rings_ring.inp", "traj_1rep.trj", counts, boxes[0], ns=1,
                    cutoffs=CUTOFFS_RING)
        write_options(run_dir / "options_ring",  do_rings=True)

    print(f"  Setup: {proto} x={x} — NA={sum(counts.values())}, box={box_avg[0]:.3f} Å")
    return run_dir, counts


def run_rings(run_dir, inp_file, opt_file, log_file):
    """Corre RINGS en run_dir con inp_file y options=opt_file."""
    env = os.environ.copy()
    cmd = [str(RINGS_BIN), inp_file]
    # RINGS lee siempre el archivo 'options' del directorio actual
    # → hacemos symlink/copia
    opt_src = run_dir / opt_file
    opt_dst = run_dir / "options"
    if opt_dst.exists() or opt_dst.is_symlink():
        opt_dst.unlink()
    opt_dst.symlink_to(opt_src.name)

    log_path = run_dir / log_file
    print(f"  Running RINGS: {inp_file} → {log_file}")
    with open(log_path, 'w') as lf:
        result = subprocess.run(cmd, cwd=run_dir, stdout=lf, stderr=subprocess.STDOUT,
                                env=env, timeout=7200)
    if result.returncode != 0:
        print(f"  WARNING: RINGS returned code {result.returncode} for {run_dir.name}")
    else:
        print(f"  OK: {run_dir.name}/{log_file}")
    opt_dst.unlink()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-rings', action='store_true',
                        help='Solo preparar g(r), sin ring statistics')
    parser.add_argument('--proto', default='all',
                        help='Protocolo: AM, IEX1, IEX2 o all')
    parser.add_argument('--setup-only', action='store_true',
                        help='Solo crear archivos, no correr RINGS')
    args = parser.parse_args()

    do_rings = not args.no_rings
    protos   = ['AM', 'IEX1', 'IEX2', 'REIEX1', 'REIEX2'] if args.proto == 'all' else [args.proto]

    for proto in protos:
        for x in PROTOCOLS[proto]['x_list']:
            print(f"\n{'='*50}")
            print(f"  {proto}  x={x}")
            result = setup_system(proto, x, do_rings=do_rings)
            if result is None or args.setup_only:
                continue
            run_dir, counts = result
            # g(r) con 3 réplicas
            run_rings(run_dir, "rings_gr.inp",   "options_gr",   "log_gr.out")
            # ring statistics con 1 réplica
            if do_rings:
                run_rings(run_dir, "rings_ring.inp", "options_ring", "log_ring.out")

    print("\nDone.")
