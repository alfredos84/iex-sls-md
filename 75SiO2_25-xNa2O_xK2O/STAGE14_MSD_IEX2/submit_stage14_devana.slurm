#!/usr/bin/env bash
#SBATCH --job-name=noca5k_msd_iex2
#SBATCH --account=p1934-26-t
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=24:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err

# TEST_GLASS_PMMCS — STAGE14 MSD IEX Proto2 (5000 oxide units, PMMCS rc8p0)
# x_parent = 0, 1, 3, 6, 9, 12 — 6 compositions × 3 replicas = 18 tasks (array 0-17)
# Layout: task i → X_VALUES[i/3], r = (i%3)+1
# Protocol: NPT heating + 1 ns equil + 4 ns production with MSD (K only)
# Temperatures: 723, 800, 1000, 1200, 1400 K  (input glasses already at 723 K)

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"
IEX2_DIR="../STAGE4_IEX_PROTO2/data/iox2_723K"

X_VALUES=(0 1 3 6 9 12)

X_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
R=$(( SLURM_ARRAY_TASK_ID % 3 + 1 ))
X=${X_VALUES[$X_IDX]}

DATAFILE="${IEX2_DIR}/IOX2_723K_xp${X}_r${R}_PMMCS_rc8p0.data"

echo "Task ${SLURM_ARRAY_TASK_ID}: x_parent=${X} r=${R}  MSD IEX2 PMMCS rc8p0"

if [[ ! -f "${DATAFILE}" ]]; then
    echo "ERROR: input not found: ${DATAFILE}"
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage14_msd_iex2.in \
    -var val_x    "${X}" \
    -var val_r    "${R}" \
    -var potfile  "${POTFILE}" \
    -var datafile "${DATAFILE}"

echo "Done: x_parent=${X} r=${R}"
