#!/usr/bin/env bash
#SBATCH --job-name=stage15_am_npt723k
#SBATCH --partition=cpu          # Change to 'gpu' if running GPU workloads
#SBATCH --nodelist=ai1
#SBATCH --array=0-20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# TEST_GLASS_PMMCS — STAGE15 AM NPT 723K (5000 oxide units, PMMCS rc8p0)
# x = 0, 1, 3, 6, 9, 12, 15 — 7 compositions × 3 replicas = 21 tasks (array 0-20)
# Layout: task i → X_VALUES[i/3], r = (i%3)+1
# Protocol: NVT 4000 ps + NPT 6000 ps at 723 K — AM reference glass equilibration
#
# Estimated walltime: ~2.3 h/run  (scaled from FG benchmark:
#   580k steps/29k atoms/80 min × (2M steps / 0.5× atoms) = 1.72×)

unset OPAL_PREFIX

mkdir -p logs

module use ~/modulefiles
module load lammps/2025.7

export LMP_BIN=$(which lmp)

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"
AM_DIR="../STAGE1_MELTQUENCH/data/asmelted_723K"

X_VALUES=(0 1 3 6 9 12 15)

X_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
R=$(( SLURM_ARRAY_TASK_ID % 3 + 1 ))
X=${X_VALUES[$X_IDX]}

DATAFILE="${AM_DIR}/AsMelted_723K_x${X}_r${R}_PMMCS_rc8p0.data"

echo "=================================================="
echo "  STAGE15 AM NPT 723K — PMMCS rc8p0"
echo "  Cluster:  $(hostname -f)"
echo "  Job:      ${SLURM_JOB_ID}  task ${SLURM_ARRAY_TASK_ID}"
echo "  x=${X}  r=${R}"
echo "  Cores:    ${SLURM_NTASKS}"
echo "  Start:    $(date)"
echo "=================================================="

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

echo "Done: x=${X} r=${R}  $(date)"
