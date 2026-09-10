#!/usr/bin/env bash
#SBATCH --job-name=noca5k_s7_iex1
#SBATCH --account=p1934-26-t
#SBATCH --array=0-17
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err
#SBATCH --mail-user=alfredo.daniel.sanchez@gmail.com
#SBATCH --mail-type=END,FAIL

# Ca-libre — STAGE7 Quench IEX Proto1 723K→300K at 1K/ps (PMMCS rc8p0)
# x_target = 1, 5, 10, 15, 20, 25  —  6 compositions × 3 replicas = 18 tasks (array 0-17)

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"
X_VALUES=(1 5 10 15 20 25)

X_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
R=$(( SLURM_ARRAY_TASK_ID % 3 + 1 ))
X=${X_VALUES[$X_IDX]}

TAG="IEX1_xt${X}_r${R}"
DATAFILE_IN="../STAGE3_IEX_PROTO1/data/iox1_723K/IOX1_723K_xt${X}_r${R}_PMMCS_rc8p0.data"
DATAFILE_OUT="data/iex1_300K/IEX1_300K_xt${X}_r${R}_PMMCS_rc8p0.data"

echo "Task ${SLURM_ARRAY_TASK_ID}: x_target=${X} r=${R}  PMMCS rc8p0  [IEX Proto1]"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p data/iex1_300K logs trajectories

mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage7_quench.in \
    -var val_datafile  "${DATAFILE_IN}" \
    -var val_tag       "${TAG}" \
    -var val_outfile   "${DATAFILE_OUT}" \
    -var potfile       "${POTFILE}"

[ $? -ne 0 ] && { echo "ERROR: xt=${X} r=${R}" >&2; exit 1; }
echo "Done: ${TAG}"
