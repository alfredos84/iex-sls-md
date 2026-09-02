#!/usr/bin/env bash
#SBATCH --job-name=noca5k_el8_iex2
#SBATCH --account=p1946-26-t
#SBATCH --array=0-17
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err

# TEST_GLASS_PMMCS — STAGE8 Athermal elastic tensor, IEX Proto2 @300K (PMMCS rc8p0)
# x_parent = 0, 1, 3, 6, 9, 12 — 6 compositions × 3 replicas = 18 tasks (array 0-17)
# Layout: task i → X_VALUES[i/3], r = (i%3)+1

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"
STRAIN_UP="5.0e-4"
X_VALUES=(0 1 3 6 9 12)

X_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
R=$(( SLURM_ARRAY_TASK_ID % 3 + 1 ))
X=${X_VALUES[$X_IDX]}
TAG="IEX2_300K_xp${X}_r${R}"
DATAFILE="../STAGE7_QUENCH_300K/data/iex2_300K/IEX2_300K_xp${X}_r${R}_PMMCS_rc8p0.data"

echo "Task ${SLURM_ARRAY_TASK_ID}: tag=${TAG}  strain_up=${STRAIN_UP}"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p results

echo "  [1/14] Reference minimize..."
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in elastic_T0_ref.in \
    -var val_datafile "${DATAFILE}" \
    -var val_tag      "${TAG}" \
    -var potfile      "${POTFILE}"
[ $? -ne 0 ] && { echo "ERROR: ref failed ${TAG}" >&2; exit 1; }

STEP=2
for DIR in 1 2 3 4 5 6; do
    for SIGN in -1 1; do
        echo "  [${STEP}/14] dir=${DIR} sign=${SIGN}..."
        mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
            -in elastic_T0_deform.in \
            -var val_tag   "${TAG}" \
            -var potfile   "${POTFILE}" \
            -var strain_up "${STRAIN_UP}" \
            -var val_dir   "${DIR}" \
            -var val_sign  "${SIGN}"
        [ $? -ne 0 ] && { echo "ERROR: deform failed ${TAG} dir=${DIR} sign=${SIGN}" >&2; exit 1; }
        STEP=$(( STEP + 1 ))
    done
done

echo "  [14/14] Computing Cij..."
python3 compute_Cij_300K.py --tag "${TAG}" --results results/
echo "Done: ${TAG}"
