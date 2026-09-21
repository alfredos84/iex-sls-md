#!/usr/bin/env bash
#SBATCH --job-name=pipeline_cao_x0_r1
#SBATCH --partition=cpu
#SBATCH --nodelist=ai2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --time=36:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Pipeline completo:
#   0) create_slab.in      -> slab.dat                    (rapido)
#   1) replicate_slab.in   -> slab_2x2.data              (rapido)
#   2) relax_surface.in    -> slab_2x2_relaxed.data       (~920 ps, largo)
#   3) create_slab_tip.in  -> slab2x2_tip.data             (rapido)
#   4) indent.in           -> indentacion completa (~320 ps)

unset OPAL_PREFIX

mkdir -p logs

module use ~/modulefiles
module load lammps/2025.7

export LMP_BIN=$(which lmp)

cd "${SLURM_SUBMIT_DIR}"

echo "=================================================="
echo "  Pipeline indentacion CaO x=0 r=1"
echo "  Cluster:  $(hostname -f)"
echo "  Job:      ${SLURM_JOB_ID}"
echo "  Cores:    ${SLURM_NTASKS}"
echo "  Start:    $(date)"
echo "=================================================="

# check_step: valida CODIGO DE SALIDA real de mpirun + que el archivo de
# salida exista. Un ERROR de LAMMPS no siempre deja al archivo ausente
# (puede quedar un archivo VIEJO de una corrida anterior) -- por eso no
# alcanza con "[[ -f archivo ]]" solo, hay que chequear $? tambien.
check_step() {
    local rc=$1 step=$2 outfile=$3
    if [[ $rc -ne 0 ]] || ! [[ -f "$outfile" ]]; then
        echo "FALLO: ${step} termino con codigo ${rc} o no genero ${outfile}"
        exit 1
    fi
}

echo "### STEP 0: create_slab.in ###"
mpirun -np "${SLURM_NTASKS}" --bind-to none "${LMP_BIN}" -in create_slab.in -log log.0_create_slab
check_step $? "create_slab.in" slab.dat

echo "### STEP 1: replicate_slab.in ###"
mpirun -np "${SLURM_NTASKS}" --bind-to none "${LMP_BIN}" -in replicate_slab.in -log log.1_replicate
check_step $? "replicate_slab.in" slab_2x2.data

echo "### STEP 2: relax_surface.in ###"
mpirun -np "${SLURM_NTASKS}" --bind-to none "${LMP_BIN}" -in relax_surface.in -log log.2_relax
check_step $? "relax_surface.in" slab_2x2_relaxed.data

echo "### STEP 3: create_slab_tip.in ###"
mpirun -np "${SLURM_NTASKS}" --bind-to none "${LMP_BIN}" -in create_slab_tip.in -log log.3_tip
check_step $? "create_slab_tip.in" slab2x2_tip.data

echo "### STEP 4: indent.in ###"
mpirun -np "${SLURM_NTASKS}" --bind-to none "${LMP_BIN}" -in indent.in -log log.4_indent
check_step $? "indent.in" tip_force_vs_disp.dat

echo "Pipeline terminado: $(date)"
