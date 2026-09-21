#!/usr/bin/env bash
#SBATCH --job-name=dryrun_noca_devana
#SBATCH --account=p1934-26-t
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"
rm -f slab.dat slab_2x2.data slab_2x2_relaxed.data slab2x2_tip.data

echo "### STEP0 ###"
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" -in create_slab.in -log dryrun.0
[[ -f slab.dat ]] || { echo "FALLO STEP0"; exit 1; }

echo "### STEP1 ###"
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" -in replicate_slab.in -log dryrun.1
[[ -f slab_2x2.data ]] || { echo "FALLO STEP1"; exit 1; }

cp slab_2x2.data slab_2x2_relaxed.data

echo "### STEP3 ###"
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" -in create_slab_tip.in -log dryrun.3
[[ -f slab2x2_tip.data ]] || { echo "FALLO STEP3"; exit 1; }

echo "DRYRUN OK"
