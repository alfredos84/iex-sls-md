#!/usr/bin/env bash
#SBATCH --job-name=tgpmmcs5k_s7_iex2
#SBATCH --account=p1934-26-t
#SBATCH --array=0-17
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err

# TEST_GLASS_PMMCS — STAGE7 Quench IEX Proto2 723K→300K at 1K/ps (PMMCS rc8p0)
# x_parent = 0, 1, 3, 6, 9, 12 — 6 compositions × 3 replicas = 18 tasks (array 0-17)
# Layout: task i → X_VALUES[i/3], r = (i%3)+1

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"
X_VALUES=(0 1 3 6 9 12)

X_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
R=$(( SLURM_ARRAY_TASK_ID % 3 + 1 ))
X=${X_VALUES[$X_IDX]}

TAG="IEX2_xp${X}_r${R}"
DATAFILE_IN="../STAGE4_IEX_PROTO2/data/iox2_723K/IOX2_723K_xp${X}_r${R}_PMMCS_rc8p0.data"
DATAFILE_OUT="data/iex2_300K/IEX2_300K_xp${X}_r${R}_PMMCS_rc8p0.data"

echo "Task ${SLURM_ARRAY_TASK_ID}: x_parent=${X} r=${R}  PMMCS rc8p0  [IEX Proto2]"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p data/iex2_300K logs

mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage7_quench.in \
    -var val_datafile  "${DATAFILE_IN}" \
    -var val_tag       "${TAG}" \
    -var val_outfile   "${DATAFILE_OUT}" \
    -var potfile       "${POTFILE}"

[ $? -ne 0 ] && { echo "ERROR: xp=${X} r=${R}" >&2; exit 1; }
echo "Done: ${TAG}"
