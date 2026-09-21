#!/usr/bin/env bash
#SBATCH --job-name=resume_indent_cao
#SBATCH --partition=cpu
#SBATCH --nodelist=ai2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Retoma SOLO indent.in -- slab2x2_tip.data ya existe (superficie
# reconstruida + punta agregada, corrida anterior cortada por time limit
# durante la indentacion misma, no hace falta repetir nada previo).

unset OPAL_PREFIX
mkdir -p logs
module use ~/modulefiles
module load lammps/2025.7
LMP_BIN=$(which lmp)

cd "${SLURM_SUBMIT_DIR}"

[[ -f slab2x2_tip.data ]] || { echo "FALLO: no existe slab2x2_tip.data"; exit 1; }

echo "### indent.in (resume) ###"
mpirun -np "${SLURM_NTASKS}" --bind-to none "${LMP_BIN}" -in indent.in -log log.4_indent
rc=$?
if [[ $rc -ne 0 ]] || ! [[ -f tip_force_vs_disp.dat ]]; then
    echo "FALLO: indent.in termino con codigo ${rc} o no genero tip_force_vs_disp.dat"
    exit 1
fi

echo "Indentacion (resume) terminada: $(date)"
