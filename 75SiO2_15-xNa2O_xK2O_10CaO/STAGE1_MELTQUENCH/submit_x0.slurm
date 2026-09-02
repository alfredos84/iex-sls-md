#!/usr/bin/env bash
#SBATCH --job-name=tgpmmcs5k_s1_x0
#SBATCH --account=p1934-26-t
#SBATCH --array=0-2
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err

# TEST_GLASS_PMMCS — STAGE1 Melt-quench (5000 oxide units, PMMCS rc8p0)
# x=0 only — 3 replicas (array tasks 0-2)

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"
X=0
DENSITY=2.482
R=$(( SLURM_ARRAY_TASK_ID + 1 ))

echo "Task ${SLURM_ARRAY_TASK_ID}: x=${X} r=${R}  density=${DENSITY}  PMMCS rc8p0"

cd "${SLURM_SUBMIT_DIR}"

mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage1_meltquench.in \
    -var val_x          "${X}" \
    -var val_r          "${R}" \
    -var target_density "${DENSITY}" \
    -var potfile        "${POTFILE}"

echo "Done: x=${X} r=${R}"
