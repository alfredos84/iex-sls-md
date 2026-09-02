#!/usr/bin/env bash
#SBATCH --job-name=tgpmmcs5k_reiex1b
#SBATCH --account=p1934-26-t
#SBATCH --array=0-17
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=../cluster_logs/%x_%A_%a.out
#SBATCH --error=../cluster_logs/%x_%A_%a.err

# TEST_GLASS_PMMCS — STAGE11b: continue interrupted NPT of Reverse IEX1
# Resumes from last saved NPT frame in the trajectory file.
# Remaining time per task: 0.4–1.0 ns (~30 min–2 h on 48 cores).
# 4h time limit is generous.

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

POTFILE="../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"

X_VALUES=(1 3 6 9 12 15)

X_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
R=$(( SLURM_ARRAY_TASK_ID % 3 + 1 ))
X=${X_VALUES[$X_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: x_target=${X} r=${R}  STAGE11b continue NPT"

cd "${SLURM_SUBMIT_DIR}"

# Determine last saved timestep from the existing NPT trajectory
TRAJ="trajectories/REIEX1_NPT_723K_xt${X}_r${R}_PMMCS_rc8p0.lammpstrj"

if [[ ! -f "${TRAJ}" ]]; then
    echo "ERROR: NPT trajectory not found: ${TRAJ}"
    exit 1
fi

LAST_TS=$(grep -A1 "ITEM: TIMESTEP" "${TRAJ}" | grep -v "ITEM:" | tail -1 | tr -d ' ')
REMAINING=$(( 10000000 - LAST_TS ))

echo "  Last timestep in dump: ${LAST_TS}  →  remaining NPT steps: ${REMAINING}"

if [[ "${REMAINING}" -le 0 ]]; then
    echo "  NPT already complete — running write_data only"
    # Reuse the last dump frame to produce the data file if somehow missed
    REMAINING=0
fi

mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage11b_continue_npt.in \
    -var val_x_target   "${X}" \
    -var val_r          "${R}" \
    -var potfile        "${POTFILE}" \
    -var val_last_ts    "${LAST_TS}" \
    -var val_remaining  "${REMAINING}"

echo "Done: x_target=${X} r=${R}"
