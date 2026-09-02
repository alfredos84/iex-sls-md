"""
compute_Cij_300K.py — STAGE8_ELASTIC_T0_300K
Post-process stress files into Cij tensor (central finite differences).
Flexible tag-based interface: works for as-melted, IEX1, and IEX2 glasses.

Formula:
  C_ji = (sigma_j_neg - sigma_j_pos) / (2 * strain_up) * cfac
  cfac = 1e-4  (bar -> GPa)

Voigt mapping (LAMMPS output: pxx pyy pzz pxy pxz pyz):
  sigma1=Pxx [0], sigma2=Pyy [1], sigma3=Pzz [2],
  sigma4=Pyz [5], sigma5=Pxz [4], sigma6=Pxy [3]

Usage:
  python compute_Cij_300K.py --tag AM300K_x6_r2    [as-melted]
  python compute_Cij_300K.py --tag IEX1_300K_xt3_r1
  python compute_Cij_300K.py --tag IEX2_300K_xp0_r3
"""

import numpy as np
import sys
import argparse
from pathlib import Path

VOIGT_TO_LAMMPS = {1: 0, 2: 1, 3: 2, 4: 5, 5: 4, 6: 3}
BAR_TO_GPA = 1.0e-4


def load_stress(path):
    line = Path(path).read_text().strip().split('\n')[-1]
    return [float(v) for v in line.split()]


def build_Cij(tag, results_dir):
    results = Path(results_dir)
    C = np.zeros((6, 6))

    for d in range(1, 7):
        neg_file = results / f"elastic_T0_stress_{tag}_dir{d}_sign-1.txt"
        pos_file = results / f"elastic_T0_stress_{tag}_dir{d}_sign1.txt"

        if not neg_file.exists() or not pos_file.exists():
            print(f"  WARNING: missing stress files for dir={d}, tag={tag}",
                  file=sys.stderr)
            continue

        neg = load_stress(neg_file)
        pos = load_stress(pos_file)

        len0      = neg[6]
        delta     = abs(neg[7])
        strain_up = delta / len0

        for j_voigt in range(1, 7):
            lmp_idx = VOIGT_TO_LAMMPS[j_voigt]
            C[j_voigt - 1, d - 1] = (
                (neg[lmp_idx] - pos[lmp_idx]) / (2.0 * strain_up) * BAR_TO_GPA
            )

    return 0.5 * (C + C.T)


def voigt_averages(C):
    KV = (C[0,0]+C[1,1]+C[2,2] + 2*(C[0,1]+C[0,2]+C[1,2])) / 9.0
    GV = (C[0,0]+C[1,1]+C[2,2] - C[0,1]-C[0,2]-C[1,2]
          + 3*(C[3,3]+C[4,4]+C[5,5])) / 15.0
    try:
        S = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return KV, GV, KV, GV, 9*KV*GV/(3*KV+GV), (3*KV-2*GV)/(2*(3*KV+GV))
    KR = 1.0 / (S[0,0]+S[1,1]+S[2,2] + 2*(S[0,1]+S[0,2]+S[1,2]))
    GR = 15.0 / (4*(S[0,0]+S[1,1]+S[2,2]) - 4*(S[0,1]+S[0,2]+S[1,2])
                 + 3*(S[3,3]+S[4,4]+S[5,5]))
    K  = 0.5*(KV + KR)
    G  = 0.5*(GV + GR)
    E  = 9*K*G / (3*K + G)
    nu = (3*K - 2*G) / (2*(3*K + G))
    return KV, GV, K, G, E, nu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag',     type=str, required=True,
                        help='Run tag, e.g. AM300K_x6_r2 or IEX1_300K_xt3_r1')
    parser.add_argument('--results', type=str, default='results')
    args = parser.parse_args()

    C = build_Cij(args.tag, args.results)
    KV, GV, K, G, E, nu = voigt_averages(C)

    print(f"\n=== Elastic tensor (GPa) — PMMCS rc8p0 300K, tag={args.tag} ===")
    rows = ['xx','yy','zz','yz','xz','xy']
    print(f"{'':6s}" + "".join(f"{h:>10s}" for h in rows))
    for i in range(6):
        print(f"{rows[i]:6s}" + "".join(f"{C[i,j]:10.3f}" for j in range(6)))

    print(f"\nVoigt:  KV={KV:.2f}  GV={GV:.2f} GPa")
    print(f"VRH:    K ={K:.2f}  G ={G:.2f}  E ={E:.2f}  nu={nu:.4f} GPa")

    out = Path(args.results) / f"elastic_T0_Cij_PMMCS_rc8p0_{args.tag}.txt"
    with open(out, 'w') as f:
        f.write("C11 C22 C33 C12 C13 C23 C44 C55 C66 "
                "C14 C15 C16 C24 C25 C26 C34 C35 C36 C45 C46 C56 "
                "KV GV K G E nu\n")
        vals = [C[0,0],C[1,1],C[2,2],C[0,1],C[0,2],C[1,2],
                C[3,3],C[4,4],C[5,5],
                C[0,3],C[0,4],C[0,5],C[1,3],C[1,4],C[1,5],
                C[2,3],C[2,4],C[2,5],C[3,4],C[3,5],C[4,5],
                KV, GV, K, G, E, nu]
        f.write(" ".join(f"{v:.6f}" for v in vals) + "\n")
    print(f"\nWrote: {out}")


if __name__ == '__main__':
    main()
