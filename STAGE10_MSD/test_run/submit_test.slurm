#!/usr/bin/env bash
#SBATCH --job-name=s10_test
#SBATCH --account=p1946-26-t
#SBATCH --ntasks-per-node=48
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=../../cluster_logs/%x_%j.out
#SBATCH --error=../../cluster_logs/%x_%j.err

# Test rápido STAGE10: x=6, r=1, T=800+1000 K, eq=10 ps, prod=50 ps
# Corre en ~5 min — verifica lectura, potencial, grupos, MSD y output

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

X=6
R=1
POTFILE="../../IPs/PMMCS/PMMCS_pedone_rc8p0.mod"
DATAFILE="../../STAGE7_QUENCH_300K/data/asmelted_300K/AsMelted_300K_x${X}_r${R}_PMMCS_rc8p0.data"

echo "STAGE10 TEST: x=${X} r=${R}"
echo "Datafile: ${DATAFILE}"
echo "Potfile:  ${POTFILE}"

cd "${SLURM_SUBMIT_DIR}"

mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" \
    -in stage10_test.in \
    -var val_x    "${X}" \
    -var val_r    "${R}" \
    -var potfile  "${POTFILE}" \
    -var datafile "${DATAFILE}"

echo "Test done. Check data/msd_Na_*.dat and data/msd_K_*.dat"
