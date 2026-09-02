#!/usr/bin/env bash
#SBATCH --job-name=tgpmmcs5k_reiex1
#SBATCH --account=euhpc_b38_167
#SBATCH --partition=dcgp_usr_prod
#SBATCH --array=0-17
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err

# TEST_GLASS_PMMCS — STAGE11 Reverse IEX Protocol 1 (5000 oxide units, PMMCS rc8p0)
# x_target = 1, 3, 6, 9, 12, 15 — 6 compositions × 3 replicas = 18 tasks (array 0-17)
# Layout: task i → X_VALUES[i/3], r = (i%3)+1
# Protocol: ALL K→Na (reverse), NVT 4000 ps + NPT 6000 ps at 723 K (10 frames each)

module purge
module load profile/chem-phys
module load lammps/29aug2024

# El modulo no provee un binario "lmp": solo variantes GPU/Kokkos
# (lmp_gpu*, lmp_kokkos*), y todas quedan linkeadas a libcuda.so.1
# (via deepmd/TensorFlow) aunque el backend elegido sea OpenMP puro.
# dcgp_usr_prod no tiene GPU ni libcuda.so.1 del driver, asi que
# apuntamos al stub "compat" del modulo cuda/12.1 para resolver el
# link en tiempo de carga (no se usa GPU real en ningun momento).
LMP_BIN="lmp_kokkos_omp"
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="/leonardo/prod/opt/compilers/cuda/12.1/none/compat:${LD_LIBRARY_PATH:-}"

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"

X_VALUES=(1 3 6 9 12 15)

X_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
R=$(( SLURM_ARRAY_TASK_ID % 3 + 1 ))
X=${X_VALUES[$X_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: x_target=${X} r=${R}  Reverse IEX1 PMMCS rc8p0"

cd "${SLURM_SUBMIT_DIR}"

srun -n "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage11_reverse_iex1.in \
    -var val_x_target "${X}" \
    -var val_r        "${R}" \
    -var potfile      "${POTFILE}"

echo "Done: x_target=${X} r=${R}"
