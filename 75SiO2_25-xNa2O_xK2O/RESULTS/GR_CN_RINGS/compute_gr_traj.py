"""
compute_gr_traj.py
Calcula g(r) parciales desde trayectorias dump LAMMPS (numpy, sin RINGS).

Para IEX2 y REIEX2: lee los NPT dump files (11 frames × 3 replicas = 33 frames)
y escribe gr-{sp1}_{sp2}.dat en el mismo formato que RINGS, para que el
script de figura (fig_gr_SiNaK_reversal.py) pueda leerlos sin cambios.

Uso:
    python3 compute_gr_traj.py              # IEX2 + REIEX2, x=0,1,3,6,9,12
    python3 compute_gr_traj.py --proto IEX2 --x 6
"""

import numpy as np
import argparse
from pathlib import Path

HERE  = Path(__file__).parent
BASE  = HERE.parent.parent

TRAJ_INFO = {
    'IEX2': {
        'traj_dir': BASE / "STAGE4_IEX_PROTO2" / "trajectories",
        'traj_npt': lambda x, r: f"IOX2_NPT_723K_xp{x}_r{r}_PMMCS_rc8p0.lammpstrj",
        'x_list': [0, 1, 3, 6, 9, 12],
    },
    'REIEX2': {
        'traj_dir': BASE / "STAGE12_REVERSE_IEX2" / "trajectories",
        'traj_npt': lambda x, r: f"REIEX2_NPT_723K_xp{x}_r{r}_PMMCS_rc8p0.lammpstrj",
        'x_list': [0, 1, 3, 6, 9, 12],
    },
}

# LAMMPS type → species name
TYPE_NAME = {1: 'Si', 2: 'O', 3: 'Ca', 4: 'Na', 5: 'K'}

# Pairs a calcular (solo los necesarios para las figuras)
PAIRS = [('Si', 'Na'), ('Si', 'K')]

DR   = 0.02    # Å — resolución, igual que RINGS
R_MAX = 8.5    # Å — máximo a calcular (figura muestra hasta 8 Å)


# ── Leer dump ────────────────────────────────────────────────────────────────

def read_lammps_dump(path):
    """
    Lee todos los frames de un dump LAMMPS.
    Devuelve lista de (box[3], {type_id: positions_array}).
    """
    frames   = []
    box      = np.zeros(3)
    rows     = []
    col_type = col_x = col_y = col_z = None
    state    = 'idle'

    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if line == 'ITEM: TIMESTEP':
                if rows:
                    frames.append(_build_frame(box, rows, col_type, col_x, col_y, col_z))
                box, rows = np.zeros(3), []
                state, box_idx = 'skip_ts', 0
            elif state == 'skip_ts':
                state = 'idle'
            elif line == 'ITEM: NUMBER OF ATOMS':
                state = 'natoms'
            elif state == 'natoms':
                state = 'box'
                box_idx = 0
            elif line.startswith('ITEM: BOX BOUNDS'):
                state, box_idx = 'box', 0
            elif state == 'box':
                lo, hi = map(float, line.split()[:2])
                box[box_idx] = hi - lo
                box_idx += 1
                if box_idx == 3:
                    state = 'expect_atoms'
            elif line.startswith('ITEM: ATOMS'):
                cols = line.split()[2:]
                col_type = cols.index('type')
                col_x    = cols.index('x')
                col_y    = cols.index('y')
                col_z    = cols.index('z')
                state    = 'atoms'
            elif state == 'atoms' and line:
                rows.append(line.split())
    if rows:
        frames.append(_build_frame(box, rows, col_type, col_x, col_y, col_z))
    return frames


def _build_frame(box, rows, ct, cx, cy, cz):
    """Convierte filas de texto a dict {type_int: xyz_array}."""
    arr = np.array(rows, dtype=float)
    types = arr[:, ct].astype(int)
    xyz   = arr[:, [cx, cy, cz]]
    by_type = {}
    for t in np.unique(types):
        by_type[t] = xyz[types == t]
    return box.copy(), by_type


# ── g(r) parcial ─────────────────────────────────────────────────────────────

def partial_gr(frames, sp1_name, sp2_name, dr=DR, r_max=R_MAX, batch=1000):
    """
    Calcula g(r) entre especies sp1 y sp2 promediando sobre todos los frames.
    Devuelve (r_centers, gr) o (None, None) si alguna especie no existe.
    batch: tamaño de chunk para pares grandes (evita OOM en O-O, etc.)
    """
    name_to_type = {v: k for k, v in TYPE_NAME.items()}
    t1 = name_to_type.get(sp1_name)
    t2 = name_to_type.get(sp2_name)
    if t1 is None or t2 is None:
        return None, None

    n_bins  = int(r_max / dr) + 1
    bins    = np.linspace(0, r_max + dr, n_bins + 1)
    r_cen   = 0.5 * (bins[:-1] + bins[1:])
    shell_vol = (4.0 / 3.0) * np.pi * (bins[1:] ** 3 - bins[:-1] ** 3)

    hist_acc = np.zeros(n_bins)
    norm_acc = np.zeros(n_bins)
    n_frames_used = 0

    for box, by_type in frames:
        pos1 = by_type.get(t1)
        pos2 = by_type.get(t2)
        if pos1 is None or pos2 is None or len(pos1) == 0 or len(pos2) == 0:
            continue

        N1, N2 = len(pos1), len(pos2)
        V     = box[0] * box[1] * box[2]
        rho2  = N2 / V
        same  = (t1 == t2)

        # Procesar en chunks de pos1 para no superar memoria
        for i0 in range(0, N1, batch):
            chunk = pos1[i0:i0+batch]
            diff  = chunk[:, np.newaxis, :] - pos2[np.newaxis, :, :]
            diff -= box * np.round(diff / box)
            dist  = np.sqrt((diff ** 2).sum(axis=2)).ravel()
            if same:
                dist = dist[dist > 1e-6]
            h, _ = np.histogram(dist, bins=bins)
            hist_acc += h

        norm_acc += len(pos1) * rho2 * shell_vol
        n_frames_used += 1

    if n_frames_used == 0:
        return None, None

    with np.errstate(divide='ignore', invalid='ignore'):
        gr = np.where(norm_acc > 0, hist_acc / norm_acc, 0.0)

    return r_cen, gr


# ── Escritura en formato RINGS ────────────────────────────────────────────────

def write_gr_file(path, r, gr, sp1, sp2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(f"# r/g(r)/dn[At1,At2]:      g(r)[{sp1},{sp2}]            dn(r)\n")
        for ri, gi in zip(r, gr):
            f.write(f"  {ri:12.3f}  {gi:20.10f}  {0.0:20.10f}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def process(proto, x):
    info    = TRAJ_INFO[proto]
    traj_dir = info['traj_dir']
    npt_fn   = info['traj_npt']

    # Leer todos los frames NPT de las 3 réplicas
    all_frames = []
    for r in (1, 2, 3):
        p = traj_dir / npt_fn(x, r)
        if not p.exists():
            print(f"  MISSING: {p.name}")
            continue
        frames = read_lammps_dump(p)
        all_frames.extend(frames)

    if not all_frames:
        print(f"  No frames for {proto} x={x}")
        return

    print(f"  {proto} x={x}: {len(all_frames)} frames", end='', flush=True)

    out_dir = HERE / f"{proto}_x{x}" / "gr" / "gr-p"
    out_dir.mkdir(parents=True, exist_ok=True)

    for sp1, sp2 in PAIRS:
        r, gr = partial_gr(all_frames, sp1, sp2)
        if r is None:
            print(f"  (skipping {sp1}-{sp2}: species absent)", end='')
            continue
        fname = out_dir / f"gr-{sp1}_{sp2}.dat"
        write_gr_file(fname, r, gr, sp1, sp2)

    print(f"  → {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', default='all', help='IEX2, REIEX2, o all')
    parser.add_argument('--x',    type=int, default=None, help='Composicion especifica')
    args = parser.parse_args()

    protos = ['IEX2', 'REIEX2'] if args.proto == 'all' else [args.proto]

    for proto in protos:
        x_list = [args.x] if args.x is not None else TRAJ_INFO[proto]['x_list']
        print(f"\n{'='*50}\n  {proto}")
        for x in x_list:
            process(proto, x)

    print("\nDone.")
