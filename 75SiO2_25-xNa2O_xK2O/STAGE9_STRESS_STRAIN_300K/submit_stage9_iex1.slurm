#!/usr/bin/env bash
#SBATCH --job-name=noca5k_s9_iex1
#SBATCH --account=p1946-26-t
#SBATCH --array=0-17
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err

# TEST_GLASS_PMMCS — STAGE9 Stress-strain, IEX Proto1 @300K (PMMCS rc8p0)
# x_target = 1, 3, 6, 9, 12, 15 — 6 compositions × 3 replicas = 18 tasks (array 0-17)
# Cada tarea ejecuta las 3 direcciones (x, y, z) en serie.

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"
X_VALUES=(1 3 6 9 12 15)

X_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
R=$(( SLURM_ARRAY_TASK_ID % 3 + 1 ))
X=${X_VALUES[$X_IDX]}
TAG="IEX1_300K_xt${X}_r${R}"
DATAFILE="../STAGE7_QUENCH_300K/data/iex1_300K/IEX1_300K_xt${X}_r${R}_PMMCS_rc8p0.data"

SEED_X=( 0 42345 22345 33345 )
SEED_Y=( 0 32345 52345 72345 )
SEED_Z=( 0 12345 22345 32345 )

echo "Task ${SLURM_ARRAY_TASK_ID}: tag=${TAG}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p results logs

echo "  [x] deformación..."
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage9_deform_x.in \
    -var val_datafile "${DATAFILE}" \
    -var val_tag      "${TAG}" \
    -var val_seed     "${SEED_X[$R]}" \
    -var potfile      "${POTFILE}"
[ $? -ne 0 ] && { echo "ERROR: x failed ${TAG}" >&2; exit 1; }

echo "  [y] deformación..."
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage9_deform_y.in \
    -var val_datafile "${DATAFILE}" \
    -var val_tag      "${TAG}" \
    -var val_seed     "${SEED_Y[$R]}" \
    -var potfile      "${POTFILE}"
[ $? -ne 0 ] && { echo "ERROR: y failed ${TAG}" >&2; exit 1; }

echo "  [z] deformación..."
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage9_deform_z.in \
    -var val_datafile "${DATAFILE}" \
    -var val_tag      "${TAG}" \
    -var val_seed     "${SEED_Z[$R]}" \
    -var potfile      "${POTFILE}"
[ $? -ne 0 ] && { echo "ERROR: z failed ${TAG}" >&2; exit 1; }

echo "Done: ${TAG}"
