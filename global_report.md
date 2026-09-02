# Global Report — IEX-SLS MD Project

**System:** 75SiO₂·(15−x)Na₂O·xK₂O·10CaO (x = 0, 1, 3, 6, 9, 12, 15 mol%)
**Force field:** PMMCS, rc = 8 Å
**Last updated:** 2026-09-02

---

## Stage status

| Stage | Description | Status | Notes |
|---|---|---|---|
| STAGE1_MELTQUENCH | Melt-quench (AM glass) | ✅ Complete | 7 compositions × 3 replicas = 21 runs |
| STAGE2_ELASTIC_T0 | Born elastic — AM 723K | ✅ Complete | |
| STAGE3_IEX_PROTO1 | IEX Proto1 (gradual Na→K) | ✅ Complete | |
| STAGE4_IEX_PROTO2 | IEX Proto2 (instantaneous Na→K) | ✅ Complete | NPT trajectories at 723K available |
| STAGE5_ELASTIC_T0 | Elastic — IEX1 723K | ✅ Complete | |
| STAGE6_ELASTIC_T0 | Elastic — IEX2 723K | ✅ Complete | |
| STAGE7_QUENCH_300K | Quench IEX glasses to 300K | ✅ Complete | Data files only (no trajectories) |
| STAGE8_ELASTIC_T0 | Born elastic — 300K | ✅ Complete | |
| STAGE9_STRESS_STRAIN | Young's modulus | ✅ Complete | |
| STAGE10_MSD | MSD Na+K at 4 temperatures | ✅ Complete | T = 723, 800, 1000, 1200 K |
| STAGE11_REVERSE_IEX1 | Reverse IEX1 (K→Na) | ✅ Complete | |
| STAGE12_REVERSE_IEX2 | Reverse IEX2 (K→Na) | ✅ Complete | NPT trajectories at 723K available |
| STAGE13_MSD_IEX1 | MSD after IEX Proto1 | ✅ Complete | |
| STAGE14_MSD_IEX2 | MSD after IEX Proto2 | ✅ Complete | |
| STAGE15_AM_NPT723K | AM glass NPT at 723K | ⏳ Pending | Slurm files ready; not yet submitted |

---

## Analysis completed

### Structural (g(r) and CN(r))

- **AM glass at 300K** — all 15 partial pairs (Si-Si, Si-O, Si-Ca, Si-Na, Si-K, O-O, O-Ca, O-Na, O-K, Ca-Ca, Ca-Na, Ca-K, Na-Na, Na-K, K-K)
  - Source: STAGE1 trajectories, 33 frames per composition (11 frames × 3 replicas)
  - Script: `RESULTS/GR_CN_RINGS/fig_gr_cn_AM.py`
  - Output: `RESULTS/fig_gr_cn_AM/fig_gr_cn_AM_{sp1}_{sp2}.pdf` (15 figures)

- **IEX2 at 723K** — Si-Na and Si-K pairs
  - Source: STAGE4 NPT trajectories, 33 frames per composition
  - Script: `RESULTS/GR_CN_RINGS/compute_gr_traj.py`
  - Output: `RESULTS/GR_CN_RINGS/IEX2_x{x}/gr/gr-p/`

- **REIEX2 at 723K** — Si-Na and Si-K pairs
  - Source: STAGE12 NPT trajectories, 33 frames per composition
  - Script: `RESULTS/GR_CN_RINGS/compute_gr_traj.py`
  - Output: `RESULTS/GR_CN_RINGS/REIEX2_x{x}/gr/gr-p/`

### Diffusion and Arrhenius

- Self-diffusion coefficients D_Na, D_K at T = 723, 800, 1000, 1200 K for x = 1, 3, 6, 9, 12
- Nernst-Planck interdiffusion coefficient D̃_Na-K
- Arrhenius activation energies Ea for all species and compositions
- Script: `RESULTS/fig_arrhenius_MSD.py`
- Output: `RESULTS/fig_arrhenius_MSD.pdf`, `RESULTS/fig_Ea_vs_x.pdf`

### Mechanical properties

- Born elastic tensor → C11, C12, C44 → E, G, ν, K (Voigt/Reuss/Hill averages)
- Young's modulus from stress-strain curves
- Scripts: `RESULTS/fig_elastic_*.py`, `RESULTS/fig_young_*.py`

### Structural indicators

- Molar volume vs x: `RESULTS/fig_molar_volume*.py`
- Number density vs x: `RESULTS/fig_density_vs_x.py`
- Qn species (Si speciation): `RESULTS/fig_Qn.py`
- LNDC: `RESULTS/fig_LNDC_IEX*.py`

---

## Key results (summary)

- **Ea(D_Na)** increases with x (more K → harder for Na to diffuse): ~0.49 → 0.52 eV
- **Ea(D_K)** decreases with x (more K → collective K dynamics easier): ~0.47 → 0.41 eV
- **Ea(D̃_Na-K)** Nernst-Planck shows non-monotonic behavior
- **IEX2 is structural reference**: instantaneous Na→K exchange captures IEX effect cleanly
- **REIEX2 recovers AM structure**: reversal to original composition restores structural correlations

---

## Pending work

- [ ] Run STAGE15 (AM NPT 723K) on a cluster to get AM reference at 723K for structural comparison
- [ ] Compute g(r)/CN(r) for all pairs at 723K (IEX2, REIEX2, and AM via STAGE15)
- [ ] Ring statistics (RINGS) for AM, IEX2, REIEX2
- [ ] Bond angle distributions
- [ ] MSD analysis post-IEX (STAGE13, STAGE14 — data available, analysis pending)

---

## Repository notes

- Repo: https://github.com/alfredos84/iex-sls-md
- Tracked: LAMMPS inputs, Slurm scripts, Python analysis scripts, potential files, figures
- Not tracked: trajectories (*.lammpstrj), data files, g(r)/CN(r) raw data, cluster logs
