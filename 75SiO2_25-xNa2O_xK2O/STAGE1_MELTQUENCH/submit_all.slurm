#!/usr/bin/env bash
#SBATCH --job-name=noca5k_s1_all
#SBATCH --account=p1934-26-t
#SBATCH --array=0-26
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err

# 75SiO2.(25-x)Na2O.xK2O — STAGE1 Melt-quench (5000 oxide units, PMMCS rc8p0, no Ca)
# x = 0,1,3,6,9,12,15,20,25 — 9 compositions x 3 replicas = 27 tasks (array 0-26)
# Layout: task i → X_VALUES[i/3], replica (i%3)+1

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"

X_VALUES=(0 1 3 6 9 12 15 20 25)

declare -A DENSITY_MAP
DENSITY_MAP[0]=2.429
DENSITY_MAP[1]=2.429
DENSITY_MAP[3]=2.429
DENSITY_MAP[6]=2.429
DENSITY_MAP[9]=2.429
DENSITY_MAP[12]=2.429
DENSITY_MAP[15]=2.429
DENSITY_MAP[20]=2.429
DENSITY_MAP[25]=2.429

X_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
R=$(( SLURM_ARRAY_TASK_ID % 3 + 1 ))
X=${X_VALUES[$X_IDX]}
DENSITY=${DENSITY_MAP[$X]}

echo "Task ${SLURM_ARRAY_TASK_ID}: x=${X} r=${R}  density=${DENSITY}  PMMCS rc8p0"

cd "${SLURM_SUBMIT_DIR}"

mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage1_meltquench.in \
    -var val_x          "${X}" \
    -var val_r          "${R}" \
    -var target_density "${DENSITY}" \
    -var potfile        "${POTFILE}"

echo "Done: x=${X} r=${R}"
