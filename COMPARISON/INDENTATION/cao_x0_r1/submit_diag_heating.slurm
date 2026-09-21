#!/usr/bin/env bash
#SBATCH --job-name=diag_heating
#SBATCH --partition=cpu
#SBATCH --nodelist=ai2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

unset OPAL_PREFIX
mkdir -p logs
module use ~/modulefiles
module load lammps/2025.7
LMP_BIN=$(which lmp)

cd "${SLURM_SUBMIT_DIR}"

[[ -f slab.dat ]] || { echo "FALLO: no existe slab.dat"; exit 1; }

mpirun -np "${SLURM_NTASKS}" --bind-to none "${LMP_BIN}" -in diag_heating_trace.in -log log.diag_heating
echo "Done: $(date)"
