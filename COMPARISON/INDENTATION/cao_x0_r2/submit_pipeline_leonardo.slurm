#!/usr/bin/env bash
#SBATCH --job-name=indent_cao_x0_r2
#SBATCH --account=euhpc_b38_167
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alfredo.sanchez@tnuni.sk
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Pipeline indentacion IEX 300K — CaO xt15 r=1
# Bulk de entrada: STAGE7_QUENCH_300K (NVT 723->300K, sin relajar estructura IEX)
#   0) create_slab.in      -> slab.dat
#   1) replicate_slab.in   -> slab_2x2.data
#   2) relax_surface.in    -> slab_2x2_relaxed.data   (230 ps, cascara 7A)
#   3) create_slab_tip.in  -> slab2x2_tip.data
#   4) indent.in           -> indentacion completa (320 ps)

module purge
module load profile/chem-phys
module load lammps/29aug2024
export OMP_NUM_THREADS=1

LMP_BIN=$HOME/local/bin/lmp

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
echo "  Pipeline indentacion CaO IEX xt15 r=1 — 300K (Leonardo)"
echo "  Job:      ${SLURM_JOB_ID}"
echo "  Nodos:    ${SLURM_NNODES}   Cores: ${SLURM_NTASKS}"
echo "  Start:    $(date)"
echo "=================================================="

echo "### STEP 0: create_slab.in ###"
mpirun --bind-to none -np "${SLURM_NTASKS}" "${LMP_BIN}" -in create_slab.in -log log.0_create_slab
check_step $? "create_slab.in" slab.dat

echo "### STEP 1: replicate_slab.in ###"
mpirun --bind-to none -np "${SLURM_NTASKS}" "${LMP_BIN}" -in replicate_slab.in -log log.1_replicate
check_step $? "replicate_slab.in" slab_2x2.data

echo "### STEP 2: relax_surface.in ###"
mpirun --bind-to none -np "${SLURM_NTASKS}" "${LMP_BIN}" -in relax_surface.in -log log.2_relax
check_step $? "relax_surface.in" slab_2x2_relaxed.data

echo "### STEP 3: create_slab_tip.in ###"
mpirun --bind-to none -np "${SLURM_NTASKS}" "${LMP_BIN}" -in create_slab_tip.in -log log.3_tip
check_step $? "create_slab_tip.in" slab2x2_tip.data

echo "### STEP 4: indent.in ###"
mpirun --bind-to none -np "${SLURM_NTASKS}" "${LMP_BIN}" -in indent.in -log log.4_indent
check_step $? "indent.in" tip_force_vs_disp.dat

echo "Pipeline terminado: $(date)"
