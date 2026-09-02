#!/usr/bin/env bash
#SBATCH --job-name=tgpmmcs5k_iex1
#SBATCH --account=p1934-26-t
#SBATCH --array=0-17
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err

# TEST_GLASS_PMMCS — STAGE3 IEX Protocol 1 (5000 oxide units, PMMCS rc8p0)
# x_target = 1, 3, 6, 9, 12, 15 — 6 compositions × 3 replicas = 18 tasks (array 0-17)
# Layout: task i → X_VALUES[i/3], r = (i%3)+1
# Protocol: NVT 4000 ps + NPT 6000 ps at 723 K (no minimization, 20 frames total)

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"

X_VALUES=(1 3 6 9 12 15)

X_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
R=$(( SLURM_ARRAY_TASK_ID % 3 + 1 ))
X=${X_VALUES[$X_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: x_target=${X} r=${R}  PMMCS rc8p0"

cd "${SLURM_SUBMIT_DIR}"

mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage3_iex_proto1.in \
    -var val_x_target "${X}" \
    -var val_r        "${R}" \
    -var potfile      "${POTFILE}"

echo "Done: x_target=${X} r=${R}"
