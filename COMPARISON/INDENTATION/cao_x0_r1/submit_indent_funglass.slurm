#!/usr/bin/env bash
#SBATCH --job-name=indent_cao_x0_r1
#SBATCH --partition=cpu
#SBATCH --nodelist=ai2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Prueba piloto de nanoindentacion — CaO x=0 r=1 (protocolo Pedone)
# Slab 2x2 + tip conico C (r=40A, h=52.5A), NVT 300K
# equilibracion 20 ps -> carga 5nm/100ps -> hold 100ps -> descarga 100ps

unset OPAL_PREFIX

mkdir -p logs

module use ~/modulefiles
module load lammps/2025.7

export LMP_BIN=$(which lmp)

cd "${SLURM_SUBMIT_DIR}"

echo "=================================================="
echo "  Indentacion CaO x=0 r=1 (piloto)"
echo "  Cluster:  $(hostname -f)"
echo "  Job:      ${SLURM_JOB_ID}"
echo "  Cores:    ${SLURM_NTASKS}"
echo "  Start:    $(date)"
echo "=================================================="

mpirun -np "${SLURM_NTASKS}" --bind-to none "${LMP_BIN}" -in indent.in

echo "Done: $(date)"
