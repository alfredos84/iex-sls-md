#!/usr/bin/env bash
#SBATCH --job-name=check_gpu
#SBATCH --partition=cpu
#SBATCH --nodelist=ai1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module use ~/modulefiles
module load lammps/2025.7
LMP_BIN=$(which lmp)
echo "LMP_BIN=$LMP_BIN"
echo "--- Installed packages ---"
"${LMP_BIN}" -h 2>&1 | sed -n '/Installed packages/,/^$/p'
echo "--- pair styles relevantes ---"
"${LMP_BIN}" -h 2>&1 | grep -iE "pedone|tersoff|buck|coul/long"
