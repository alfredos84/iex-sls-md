"""
structure_analysis.py
Shared routines: read LAMMPS .data, build Si-O neighbor list, compute
Qn speciation and O-Si-O / Si-O-Si bond angles.

Atom types: Si=1, O=2, Ca=3, Na=4, K=5
"""

import numpy as np
from pathlib import Path

RC_SiO = 2.1   # Å — Si-O bond cutoff


# ── Read LAMMPS .data ─────────────────────────────────────────────────────────

def read_data(path):
    """Return box [Lx,Ly,Lz] and atoms array (N,4): [type, x, y, z]."""
    box, atoms = {}, []
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
                        atoms.append([int(p[1]),
                                      float(p[3]), float(p[4]), float(p[5])])
                    except ValueError:
                        pass
    Lx = box['x'][1] - box['x'][0]
    Ly = box['y'][1] - box['y'][0]
    Lz = box['z'][1] - box['z'][0]
    return np.array([Lx, Ly, Lz]), np.array(atoms)


# ── Minimum-image displacement ─────────────────────────────────────────────────

def mic(dr, L):
    """Apply minimum image convention for orthorhombic box."""
    return dr - L * np.round(dr / L)


# ── Build Si-O neighbor list ──────────────────────────────────────────────────

def build_SiO_neighbors(box, atoms, rc=RC_SiO):
    """
    Returns:
        si_idx  : indices (into atoms) of all Si
        o_idx   : indices (into atoms) of all O
        si_neigh: list[list[int]] — for each Si, list of O atom indices bonded
        o_neigh : list[list[int]] — for each O,  list of Si atom indices bonded
    """
    L = box
    pos  = atoms[:, 1:4]
    typ  = atoms[:, 0].astype(int)

    si_idx = np.where(typ == 1)[0]
    o_idx  = np.where(typ == 2)[0]

    si_pos = pos[si_idx]
    o_pos  = pos[o_idx]

    si_neigh = [[] for _ in si_idx]
    o_neigh  = [[] for _ in o_idx]

    rc2 = rc * rc
    for ii, (si_i, spi) in enumerate(zip(si_idx, si_pos)):
        dr  = mic(o_pos - spi, L)
        d2  = (dr ** 2).sum(axis=1)
        bonded_o = np.where(d2 < rc2)[0]      # local indices into o_idx
        for jo in bonded_o:
            si_neigh[ii].append(jo)
            o_neigh[jo].append(ii)

    return si_idx, o_idx, si_neigh, o_neigh


# ── Qn speciation ─────────────────────────────────────────────────────────────

def compute_Qn(si_neigh, o_neigh):
    """
    For each Si, count bridging oxygens (BO = O bonded to ≥2 Si).
    Returns array of Q-values (0..4) for each Si.
    """
    q = []
    for ii, o_list in enumerate(si_neigh):
        n_bo = sum(1 for jo in o_list if len(o_neigh[jo]) >= 2)
        q.append(min(n_bo, 4))
    return np.array(q)


# ── Bond angles ───────────────────────────────────────────────────────────────

def compute_angles_OSiO(box, atoms, si_neigh, si_idx, o_idx):
    """O-Si-O angles: for each Si tetrahedron, all pairs of O neighbors."""
    L   = box
    pos = atoms[:, 1:4]
    o_pos = pos[o_idx]

    angles = []
    for ii, o_list in enumerate(si_neigh):
        if len(o_list) < 2:
            continue
        si_pos_i = pos[si_idx[ii]]
        o_vecs = []
        for jo in o_list:
            dv = mic(o_pos[jo] - si_pos_i, L)
            d  = np.linalg.norm(dv)
            if d > 0:
                o_vecs.append(dv / d)
        for a in range(len(o_vecs)):
            for b in range(a + 1, len(o_vecs)):
                cos_t = np.clip(np.dot(o_vecs[a], o_vecs[b]), -1, 1)
                angles.append(np.degrees(np.arccos(cos_t)))
    return np.array(angles)


def compute_angles_SiOSi(box, atoms, o_neigh, si_idx, o_idx):
    """Si-O-Si angles: only for bridging O (bonded to ≥2 Si)."""
    L   = box
    pos = atoms[:, 1:4]
    si_pos = pos[si_idx]

    angles = []
    for jo, si_list in enumerate(o_neigh):
        if len(si_list) < 2:
            continue
        o_pos_j = pos[o_idx[jo]]
        si_vecs = []
        for ii in si_list:
            dv = mic(si_pos[ii] - o_pos_j, L)
            d  = np.linalg.norm(dv)
            if d > 0:
                si_vecs.append(dv / d)
        for a in range(len(si_vecs)):
            for b in range(a + 1, len(si_vecs)):
                cos_t = np.clip(np.dot(si_vecs[a], si_vecs[b]), -1, 1)
                angles.append(np.degrees(np.arccos(cos_t)))
    return np.array(angles)


# ── Average over replicas ──────────────────────────────────────────────────────

def avg_Qn(paths):
    """Compute mean Qn fraction over multiple .data files."""
    counts = np.zeros(5)   # Q0..Q4
    total  = 0
    for p in paths:
        box, atoms = read_data(p)
        si_idx, o_idx, si_neigh, o_neigh = build_SiO_neighbors(box, atoms)
        q = compute_Qn(si_neigh, o_neigh)
        for n in range(5):
            counts[n] += (q == n).sum()
        total += len(q)
    return counts / total if total > 0 else counts


def avg_angles(paths, angle_type='OSiO'):
    """Collect all angles over multiple .data files."""
    all_angles = []
    for p in paths:
        box, atoms = read_data(p)
        si_idx, o_idx, si_neigh, o_neigh = build_SiO_neighbors(box, atoms)
        if angle_type == 'OSiO':
            a = compute_angles_OSiO(box, atoms, si_neigh, si_idx, o_idx)
        else:
            a = compute_angles_SiOSi(box, atoms, o_neigh, si_idx, o_idx)
        all_angles.append(a)
    return np.concatenate(all_angles) if all_angles else np.array([])
