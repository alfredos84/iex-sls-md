#!/usr/bin/env bash
#SBATCH --job-name=noca5k_am_npt723
#SBATCH --account=p1934-26-t
#SBATCH --array=0-26
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=04:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err

# TEST_GLASS_PMMCS — STAGE15 AM NPT 723K (5000 oxide units, PMMCS rc8p0)
# x = 0, 1, 3, 6, 9, 12, 15 — 7 compositions × 3 replicas = 21 tasks (array 0-20)
# Layout: task i → X_VALUES[i/3], r = (i%3)+1
# Protocol: NVT 4000 ps + NPT 6000 ps at 723 K — AM reference glass equilibration

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"
AM_DIR="../STAGE1_MELTQUENCH/data/asmelted_723K"

X_VALUES=(0 1 3 6 9 12 15 20 25)

X_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
R=$(( SLURM_ARRAY_TASK_ID % 3 + 1 ))
X=${X_VALUES[$X_IDX]}

DATAFILE="${AM_DIR}/AsMelted_723K_x${X}_r${R}_PMMCS_rc8p0.data"

echo "Task ${SLURM_ARRAY_TASK_ID}: x=${X} r=${R}  AM NPT 723K PMMCS rc8p0"

if [[ ! -f "${DATAFILE}" ]]; then
    echo "ERROR: input not found: ${DATAFILE}"
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage15_am_npt723k.in \
    -var val_x    "${X}" \
    -var val_r    "${R}" \
    -var potfile  "${POTFILE}"

echo "Done: x=${X} r=${R}"
