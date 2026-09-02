#!/usr/bin/env bash
#SBATCH --job-name=tgpmmcs5k_elT0_all
#SBATCH --account=p1934-26-t
#SBATCH --array=0-20
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err

# TEST_GLASS_PMMCS — STAGE2 Athermal elastic tensor (T=0 K, PMMCS rc8p0, 5000 oxide units)
# x = 0,1,3,6,9,12,15 — 7 compositions x 3 replicas = 21 tasks (array 0-20)

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"
STRAIN_UP="5.0e-4"

X_VALUES=(0 1 3 6 9 12 15)
X_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
R=$(( SLURM_ARRAY_TASK_ID % 3 + 1 ))
X=${X_VALUES[$X_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: x=${X} r=${R}  PMMCS rc8p0  strain_up=${STRAIN_UP}"

cd "${SLURM_SUBMIT_DIR}"

echo "  [1/14] Reference minimize..."
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in elastic_T0_ref.in \
    -var val_x   "${X}" \
    -var val_r   "${R}" \
    -var potfile "${POTFILE}"

[ $? -ne 0 ] && { echo "ERROR: ref failed x=${X} r=${R}" >&2; exit 1; }

STEP=2
for DIR in 1 2 3 4 5 6; do
    for SIGN in -1 1; do
        echo "  [${STEP}/14] dir=${DIR} sign=${SIGN}..."
        mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
            -in elastic_T0_deform.in \
            -var val_x     "${X}" \
            -var val_r     "${R}" \
            -var potfile   "${POTFILE}" \
            -var strain_up "${STRAIN_UP}" \
            -var val_dir   "${DIR}" \
            -var val_sign  "${SIGN}"
        [ $? -ne 0 ] && { echo "ERROR: deform failed x=${X} r=${R} dir=${DIR} sign=${SIGN}" >&2; exit 1; }
        STEP=$((STEP + 1))
    done
done

echo "  [14/14] Computing Cij tensor..."
python3 compute_Cij.py --x "${X}" --r "${R}" --results results/

echo "Done: x=${X} r=${R}"
