# CLAUDE.md — IEX_SLS project

## Language
- All conversation in **Spanish**.
- English only for figure labels, axis titles, and publication text.

## Interaction rules (MANDATORY)
- **Do nothing without being explicitly asked.**
- When investigating a problem: search, analyze, then report to the user — wait for a decision before acting.
- Never add features, error handling, or abstractions beyond what was requested.

## Project overview

**Full name:** Ion-Exchange in Soda-Lime-Silica (IEX-SLS) glass — molecular dynamics study
**System:** 75SiO₂·(15−x)Na₂O·xK₂O·10CaO (x = 0, 1, 3, 6, 9, 12, 15 mol%)
**Goal:** Simulate thermal ion exchange (Na→K) in SLS glass and study structural, mechanical, and diffusion changes.

## Force field: PMMCS (rc = 8 Å)

- Style: `hybrid/overlay pedone 8.0 coul/long 12.0` + `kspace_style pppm 1e-6`
- Potential file: `IPs/PMMCS/PMMCS_pedone_rc8p0.mod`
- Atom types: Si=1, O=2, Ca=3, Na=4, K=5
- System size: ~5000 oxide units → ~29,000 atoms (exact count depends on x)

## Compositions

| x | Na₂O (mol%) | K₂O (mol%) | Notes |
|---|---|---|---|
| 0 | 15 | 0 | No K |
| 1 | 14 | 1 | |
| 3 | 12 | 3 | |
| 6 | 9 | 6 | |
| 9 | 6 | 9 | |
| 12 | 3 | 12 | |
| 15 | 0 | 15 | No Na |

3 replicas (r=1,2,3) per composition → 21 independent simulations per stage.

## Workflow stages

| Stage | Name | Description |
|---|---|---|
| STAGE1_MELTQUENCH | Melt-quench | Build AM glass: random → melt 5000 K → quench → 723 K snapshot + 300 K trajectories |
| STAGE2_ELASTIC_T0 | Elastic AM 723K | Born elastic constants of AM glass at 723 K |
| STAGE3_IEX_PROTO1 | IEX Proto1 | Ion exchange (gradual: Na→K swaps iteratively) at 723 K |
| STAGE4_IEX_PROTO2 | IEX Proto2 | Ion exchange (instantaneous: `set type 4 type 5`, all Na→K) at 723 K |
| STAGE5_ELASTIC_T0 | Elastic IEX1 723K | Born elastic constants after IEX Proto1 |
| STAGE6_ELASTIC_T0 | Elastic IEX2 723K | Born elastic constants after IEX Proto2 |
| STAGE7_QUENCH_300K | Quench to 300K | Quench IEX glasses from 723→300 K, write data files (no trajectories) |
| STAGE8_ELASTIC_T0 | Elastic 300K | Born elastic constants at 300 K |
| STAGE9_STRESS_STRAIN | Stress-strain | Young's modulus via uniaxial deformation |
| STAGE10_MSD | MSD | Self-diffusion of Na, K at T=723,800,1000,1200 K → Arrhenius Ea |
| STAGE11_REVERSE_IEX1 | Reverse IEX1 | Reverse K→Na exchange from IEX Proto1 |
| STAGE12_REVERSE_IEX2 | Reverse IEX2 | Reverse K→Na exchange from IEX Proto2 → returns to AM composition |
| STAGE13_MSD_IEX1 | MSD IEX1 | Diffusion after IEX Proto1 |
| STAGE14_MSD_IEX2 | MSD IEX2 | Diffusion after IEX Proto2 |
| STAGE15_AM_NPT723K | AM NPT 723K | AM glass equilibrated at 723 K (NVT 1ns + NPT 1ns, 11 frames) |

### Key naming conventions

- AM glass data: `AsMelted_723K_x{x}_r{r}_PMMCS_rc8p0.data`
- IEX2 trajectories: `IOX2_NPT_723K_xp{x}_r{r}_PMMCS_rc8p0.lammpstrj`
- REIEX2 trajectories: `REIEX2_NPT_723K_xp{x}_r{r}_PMMCS_rc8p0.lammpstrj`
- AM 300K trajectories: `AsMelted_300K_x{x}_r{r}_PMMCS_rc8p0.lammpstrj` (STAGE1)
- STAGE15 trajectories: `AM_NPT_723K_x{x}_r{r}_PMMCS_rc8p0.lammpstrj`

### Snapshot counts (per composition)

| Protocol | T (K) | Frames/replica | Replicas | Total |
|---|---|---|---|---|
| AM (STAGE1) | 300 | 11 | 3 | 33 |
| IEX2 (STAGE4) | 723 | 11 | 3 | 33 |
| REIEX2 (STAGE12) | 723 | 11 | 3 | 33 |
| STAGE15 AM | 723 | 11 | 3 | 33 |

## Analysis scripts (RESULTS/)

| Script | What it does |
|---|---|
| `fig_arrhenius_MSD.py` | Self-diffusion + Arrhenius Ea for Na, K, Darken/Nernst-Planck interdiffusion |
| `fig_Ea_vs_x.py` | Ea vs x (eV and kJ/mol) — **REFERENCE FIGURE STYLE** |
| `fig_LNDC_IEX1.py / IEX2.py` | Local network distortion coefficient |
| `fig_elastic_*.py` | Elastic moduli vs x or T |
| `fig_molar_volume*.py` | Molar volume vs x |
| `fig_density_vs_x.py` | Number density vs x |
| `GR_CN_RINGS/compute_gr_traj.py` | Compute g(r) from LAMMPS dump trajectories (numpy) |
| `GR_CN_RINGS/fig_gr_cn_AM.py` | Figures g(r) + CN(r) for all pairs, AM 300K |
| `fig_gr_SiNaK_reversal.py` | Comparison AM vs IEX2 vs REIEX2 for Si-Na and Si-K |

## Figure style (from fig_Ea_vs_x.py — USE THIS AS REFERENCE)

```python
fig, ax = plt.subplots(figsize=(6.5, 5.5))   # roughly square panels
ax.plot(x, y, color=c, lw=1.8)               # single call, NO separate marker overlay
ax.grid(axis='y', lw=0.35, color='#dddddd', zorder=0)   # y-axis grid ONLY
ax.tick_params(labelsize=9.5)                # standard (outward) ticks
ax.legend(fontsize=9.5, frameon=True, framealpha=0.9, edgecolor='#cccccc')
# For curves with many points (g(r), MSD): lines only, no markers
# For scatter with few points (Ea vs x): marker= in the plot() call, no separate overlay
```

Colors per composition (sequential):
```python
X_COLORS = {0:'#1a3a5c', 1:'#1f6fa0', 3:'#2ca0c4', 6:'#4cbf8f',
            9:'#a8d05a', 12:'#e08c2a', 15:'#c0392b'}
```

## HPC clusters

### DEVANA (NSCC Slovakia)
- SSH alias: `devana` → `login.devana.nscc.sk:5522`, user `adsr1984`
- Key: `~/.ssh/id_ed25519`
- Slurm account: `p1934-26-t`
- LAMMPS binary: `$HOME/bin/lmp_2024_cpu`
- Modules: `module load foss/2021b`
- Walltime typical: 4h for most stages
- Submit: `sbatch submit_stageXX_devana.slurm`

### FunGlass (ai1/ai2, internal cluster)
- Accessible from pcHAL as `cn01` (same network) — no SSH key setup needed from HAL
- No `--account` in Slurm (not configured)
- Partition: `cpu`, exclusive
- LAMMPS: `module use ~/modulefiles && module load lammps/2025.7` → `lmp`
- Walltime: up to 72h
- Must `unset OPAL_PREFIX` before running
- Performance: ~80 min / 580,000 steps / 29,000 atoms / 48 cores
- Submit: `sbatch submit_stageXX_funglass.slurm`

### Leonardo (CINECA, Italy)
- SSH: `login.leonardo.cineca.it`, user `asanche2` (in `~/.ssh/config` as `leonardo`)
- From pcHAL: `leonardo_mount` = `sshfs leonardo:/leonardo/home/userexternal/asanche2 ~/Leonardo`
- AccountID: `euhpc_b38_167` (check availability — can be down)
- Partition: `dcgp_usr_prod`
- LAMMPS: `lmp_kokkos_omp` (Kokkos GPU build)
- Modules: `module purge; module load profile/chem-phys; module load lammps/29aug2024`
- Export: `OMP_NUM_THREADS=1`
- Submit: `sbatch submit_stageXX_leonardo.slurm`

## Machines

| Machine | OS | Role |
|---|---|---|
| pcHAL | Linux (Ubuntu) | Primary workstation — where all code runs and this session lives |
| pcFG | Windows | Secondary — has a copy of the repo, submits FunGlass jobs |

pcFG accesses the repo via GitHub. File paths use Linux conventions on pcHAL.

## Repository

- URL: https://github.com/alfredos84/iex-sls-md
- SSH remote: `git@github.com:alfredos84/iex-sls-md.git`
- What's tracked: LAMMPS inputs (*.in), Slurm scripts (*.slurm), Python scripts (*.py), potential files (*.mod), figures (PDFs/PNGs)
- What's NOT tracked: trajectory files (*.lammpstrj), LAMMPS data files, computed g(r)/CN(r) data, cluster logs, rings-code-v1.3.5/, __pycache__/
- Update policy: after each completed stage, commit and push new inputs, scripts, and figures

## Key physical parameters

- dt = 0.001 ps (metal units) for most stages; 0.002 ps for MSD stages
- NPT: P=0 bar (iso), damping 1.0 ps
- NVT: temp damping 0.1 ps
- Neighbor: 2.0 Å bin, delay 0, check every step
- PPPM accuracy: 1e-6
- Temperatures studied: 600, 723, 800, 1000, 1200 K (MSD)
- Ion exchange temperature: 723 K

## Notes for a fresh session

1. The primary working directory is `/home/alfredo/Simulations_MD_LAMMPS/IEX_SLS/`
2. All Python scripts use `Path(__file__).parent` — run them from their own directory or use absolute paths
3. g(r) is computed with `RESULTS/GR_CN_RINGS/compute_gr_traj.py` (not RINGS software, which is slow)
4. RINGS software is only used for ring statistics (rstat/)
5. CN(r) = ρ_B × 4π ∫₀ʳ g(r') r'² dr' — computed by cumsum in the analysis scripts
6. When launching jobs: user will say "lanza esto a DEVANA/FG/LEONARDO" — prepare the right slurm variant
