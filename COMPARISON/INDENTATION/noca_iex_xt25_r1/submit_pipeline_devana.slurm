#!/usr/bin/env bash
#SBATCH --job-name=indent_noca_iex_xt25_r1
#SBATCH --account=p1934-26-t
#SBATCH --partition=short
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Pipeline IEX (cascara termica delgada, 7 A, en vez de calentar toda la
# losa como en el AM -- ver relax_surface.in para el porque):
#   0) create_slab.in      -> slab.dat        (lee IOX1_723K_xt15_r1, STAGE3)
#   1) replicate_slab.in   -> slab_2x2.data
#   2) relax_surface.in    -> slab_2x2_relaxed.data   (~230 ps, cascara+core)
#   3) create_slab_tip.in  -> slab2x2_tip.data
#   4) indent.in           -> indentacion completa (~320 ps)

module load foss/2021b
LMP_BIN=$HOME/bin/lmp_2024_cpu

mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"

check_step() {
    local rc=$1 step=$2 outfile=$3
    if [[ $rc -ne 0 ]] || ! [[ -f "$outfile" ]]; then
        echo "FALLO: ${step} termino con codigo ${rc} o no genero ${outfile}"
        exit 1
    fi
}

echo "=================================================="
echo "  Pipeline indentacion sin-Ca IEX xt25 r=1 (DEVANA)"
echo "  Job:      ${SLURM_JOB_ID}"
echo "  Nodos:    ${SLURM_NNODES}   Cores: ${SLURM_NTASKS}"
echo "  Start:    $(date)"
echo "=================================================="

echo "### STEP 0: create_slab.in ###"
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" -in create_slab.in -log log.0_create_slab
check_step $? "create_slab.in" slab.dat

echo "### STEP 1: replicate_slab.in ###"
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" -in replicate_slab.in -log log.1_replicate
check_step $? "replicate_slab.in" slab_2x2.data

echo "### STEP 2: relax_surface.in ###"
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" -in relax_surface.in -log log.2_relax
check_step $? "relax_surface.in" slab_2x2_relaxed.data

echo "### STEP 3: create_slab_tip.in ###"
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" -in create_slab_tip.in -log log.3_tip
check_step $? "create_slab_tip.in" slab2x2_tip.data

echo "### STEP 4: indent.in ###"
mpirun -np "${SLURM_NTASKS}" "${LMP_BIN}" -in indent.in -log log.4_indent
check_step $? "indent.in" tip_force_vs_disp.dat

echo "Pipeline terminado: $(date)"
