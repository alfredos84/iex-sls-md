#!/usr/bin/env bash
#SBATCH --job-name=resume_cao_x0_r1
#SBATCH --partition=cpu
#SBATCH --nodelist=ai2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Retoma el pipeline desde slab_2x2_relaxed.data (ya generado, 130500 atomos,
# caja 3x3, superficie reconstruida) sin repetir la reconstruccion termica:
#   fix) fix_atom_types.in  -> repara header (6 atom types)
#   3)   create_slab_tip.in -> slab2x2_tip.data
#   4)   indent.in          -> indentacion completa

unset OPAL_PREFIX

mkdir -p logs

module use ~/modulefiles
module load lammps/2025.7

export LMP_BIN=$(which lmp)

cd "${SLURM_SUBMIT_DIR}"

echo "=================================================="
echo "  Resume pipeline CaO x=0 r=1 (caja 3x3)"
echo "  Cluster:  $(hostname -f)"
echo "  Job:      ${SLURM_JOB_ID}"
echo "  Cores:    ${SLURM_NTASKS}"
echo "  Start:    $(date)"
echo "=================================================="

echo "### FIX: fix_atom_types.in ###"
mpirun -np "${SLURM_NTASKS}" --bind-to none "${LMP_BIN}" -in fix_atom_types.in -log log.fix_types

echo "### STEP 3: create_slab_tip.in ###"
mpirun -np "${SLURM_NTASKS}" --bind-to none "${LMP_BIN}" -in create_slab_tip.in -log log.3_tip
[[ -f slab2x2_tip.data ]] || { echo "FALLO: no se genero slab2x2_tip.data"; exit 1; }

echo "### STEP 4: indent.in ###"
mpirun -np "${SLURM_NTASKS}" --bind-to none "${LMP_BIN}" -in indent.in -log log.4_indent

echo "Pipeline (resume) terminado: $(date)"
