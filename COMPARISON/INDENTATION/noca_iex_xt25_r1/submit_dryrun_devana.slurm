#!/usr/bin/env bash
#SBATCH --job-name=dryrun_iex_noca_devana
#SBATCH --account=p1934-26-t
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"
rm -f slab.dat slab_2x2.data slab_2x2_relaxed.data

echo "### STEP0: create_slab.in ###"
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" -in create_slab.in -log dryrun.0
[[ -f slab.dat ]] || { echo "FALLO STEP0"; exit 1; }

echo "### STEP1: replicate_slab.in ###"
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" -in replicate_slab.in -log dryrun.1
[[ -f slab_2x2.data ]] || { echo "FALLO STEP1"; exit 1; }

echo "### STEP2: relax_surface_test.in (6000 pasos, no los 230000 reales) ###"
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" -in relax_surface_test.in -log dryrun.2
[[ -f slab_2x2_relaxed.data ]] || { echo "FALLO STEP2"; exit 1; }

echo "DRYRUN OK"
