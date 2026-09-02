#!/usr/bin/env bash
#SBATCH --job-name=noca5k_msd_iex1
#SBATCH --account=euhpc_b38_167
#SBATCH --partition=dcgp_usr_prod
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err

# TEST_GLASS_PMMCS — STAGE13 MSD IEX Proto1 (5000 oxide units, PMMCS rc8p0)
# x_target = 1, 3, 6, 9, 12, 15 — 6 compositions × 3 replicas = 18 tasks (array 0-17)
# Layout: task i → X_VALUES[i/3], r = (i%3)+1
# Protocol: NPT heating + 1 ns equil + 4 ns production with MSD
# Temperatures: 723, 800, 1000, 1200, 1400 K  (input glasses already at 723 K)

module purge
module load profile/chem-phys
module load lammps/29aug2024

LMP_BIN="lmp_kokkos_omp"
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/leonardo/prod/opt/compilers/cuda/12.1/none/compat:${LD_LIBRARY_PATH:-}"

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"
IEX1_DIR="../STAGE3_IEX_PROTO1/data/iox1_723K"

X_VALUES=(1 3 6 9 12 15)

X_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
R=$(( SLURM_ARRAY_TASK_ID % 3 + 1 ))
X=${X_VALUES[$X_IDX]}

DATAFILE="${IEX1_DIR}/IOX1_723K_xt${X}_r${R}_PMMCS_rc8p0.data"

echo "Task ${SLURM_ARRAY_TASK_ID}: x_target=${X} r=${R}  MSD IEX1 PMMCS rc8p0"

if [[ ! -f "${DATAFILE}" ]]; then
    echo "ERROR: input not found: ${DATAFILE}"
    exit 1
fi

cd "${SLURM_SUBMIT_DIR}"

srun -n "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage13_msd_iex1.in \
    -var val_x    "${X}" \
    -var val_r    "${R}" \
    -var potfile  "${POTFILE}" \
    -var datafile "${DATAFILE}"

echo "Done: x_target=${X} r=${R}"
