#!/usr/bin/env bash
#SBATCH --job-name=dryrun_cao_iex_xt15_300K
#SBATCH --account=euhpc_b38_167
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alfredo.sanchez@tnuni.sk
#SBATCH --time=00:40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load profile/chem-phys
module load lammps/29aug2024
export OMP_NUM_THREADS=1

LMP_BIN=$HOME/local/bin/lmp

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"
rm -f slab.dat slab_2x2.data slab_2x2_relaxed.data

echo "### STEP0: create_slab.in ###"
mpirun --bind-to none -np "${SLURM_NTASKS}" "${LMP_BIN}" -in create_slab.in -log dryrun.0
[[ -f slab.dat ]] || { echo "FALLO STEP0"; exit 1; }

echo "### STEP1: replicate_slab.in ###"
mpirun --bind-to none -np "${SLURM_NTASKS}" "${LMP_BIN}" -in replicate_slab.in -log dryrun.1
[[ -f slab_2x2.data ]] || { echo "FALLO STEP1"; exit 1; }

echo "### STEP2: relax_surface_test.in (6000 pasos) ###"
mpirun --bind-to none -np "${SLURM_NTASKS}" "${LMP_BIN}" -in relax_surface_test.in -log dryrun.2
[[ -f slab_2x2_relaxed.data ]] || { echo "FALLO STEP2"; exit 1; }

echo "DRYRUN OK"
