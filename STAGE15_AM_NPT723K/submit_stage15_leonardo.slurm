#!/usr/bin/env bash
#SBATCH --job-name=tgpmmcs5k_am_npt723
#SBATCH --account=euhpc_b38_167
#SBATCH --partition=dcgp_usr_prod
#SBATCH --array=0-20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err

# TEST_GLASS_PMMCS — STAGE15 AM NPT 723K (5000 oxide units, PMMCS rc8p0)
# x = 0, 1, 3, 6, 9, 12, 15 — 7 compositions × 3 replicas = 21 tasks (array 0-20)
# Layout: task i → X_VALUES[i/3], r = (i%3)+1
# Protocol: NVT 4000 ps + NPT 6000 ps at 723 K — AM reference glass equilibration

module purge
module load profile/chem-phys
module load lammps/29aug2024

LMP_BIN="lmp_kokkos_omp"
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/leonardo/prod/opt/compilers/cuda/12.1/none/compat:${LD_LIBRARY_PATH:-}"

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"
AM_DIR="../STAGE1_MELTQUENCH/data/asmelted_723K"

X_VALUES=(0 1 3 6 9 12 15)

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

srun -n "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage15_am_npt723k.in \
    -var val_x    "${X}" \
    -var val_r    "${R}" \
    -var potfile  "${POTFILE}"

echo "Done: x=${X} r=${R}"
