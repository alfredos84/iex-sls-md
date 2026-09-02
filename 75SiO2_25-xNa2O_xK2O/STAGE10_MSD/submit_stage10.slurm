#!/usr/bin/env bash
#SBATCH --job-name=noca5k_s10_msd
#SBATCH --account=p1934-26-t
#SBATCH --array=0-14
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err

# STAGE10 — MSD Na & K at multiple T for Arrhenius (Ea IEX)
# x = 1,3,6,9,12 — 5 compositions x 3 replicas = 15 tasks (array 0-14)
# Layout: task i → X_VALUES[i/3], replica (i%3)+1

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"
AM300_DIR="../STAGE7_QUENCH_300K/data/asmelted_300K"

X_VALUES=(1 3 6 9 12)

X_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
R=$(( SLURM_ARRAY_TASK_ID % 3 + 1 ))
X=${X_VALUES[$X_IDX]}

DATAFILE="${AM300_DIR}/AsMelted_300K_x${X}_r${R}_PMMCS_rc8p0.data"

echo "Task ${SLURM_ARRAY_TASK_ID}: x=${X} r=${R}  PMMCS rc8p0"
echo "Input: ${DATAFILE}"

cd "${SLURM_SUBMIT_DIR}"

mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage10_msd.in \
    -var val_x   "${X}" \
    -var val_r   "${R}" \
    -var potfile "${POTFILE}" \
    -var datafile "${DATAFILE}"

echo "Done: x=${X} r=${R}"
