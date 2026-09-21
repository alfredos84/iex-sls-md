#!/usr/bin/env bash
#SBATCH --job-name=indent_noca_iex_xt25_r3_723K
#SBATCH --account=euhpc_b38_167
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alfredo.sanchez@tnuni.sk
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Pipeline indentacion IEX — noCa xt25 r=3
# Bulk de entrada: STAGE3_IEX_PROTO1 (IOX1_723K)
#   0) create_slab.in -> slab.dat   1) replicate_slab.in -> slab_2x2.data
#   2) relax_surface.in -> slab_2x2_relaxed.data (230 ps, cascara 7A)
#   3) create_slab_tip.in -> slab2x2_tip.data   4) indent.in (320 ps, restart cada 10000 pasos)

module purge
module load profile/chem-phys
module load lammps/29aug2024
export OMP_NUM_THREADS=1
# FFTW de compilacion (evita segfault en fftw_destroy_plan)
export LD_LIBRARY_PATH=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-12.2.0/fftw-3.3.10-f3ms2lmqzjugb43xzrxjkvmvohumyruy/lib:$LD_LIBRARY_PATH

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

echo "Pipeline noCa IEX xt25 r=3 (Leonardo) Job ${SLURM_JOB_ID} Cores ${SLURM_NTASKS} Start $(date)"
echo "### STEP 0 ###"; mpirun --bind-to none -np "${SLURM_NTASKS}" "${LMP_BIN}" -in create_slab.in -log log.0_create_slab; check_step $? create_slab.in slab.dat
echo "### STEP 1 ###"; mpirun --bind-to none -np "${SLURM_NTASKS}" "${LMP_BIN}" -in replicate_slab.in -log log.1_replicate; check_step $? replicate_slab.in slab_2x2.data
echo "### STEP 2 ###"; mpirun --bind-to none -np "${SLURM_NTASKS}" "${LMP_BIN}" -in relax_surface.in -log log.2_relax; check_step $? relax_surface.in slab_2x2_relaxed.data
echo "### STEP 3 ###"; mpirun --bind-to none -np "${SLURM_NTASKS}" "${LMP_BIN}" -in create_slab_tip.in -log log.3_tip; check_step $? create_slab_tip.in slab2x2_tip.data
echo "### STEP 4 ###"; mpirun --bind-to none -np "${SLURM_NTASKS}" "${LMP_BIN}" -in indent.in -log log.4_indent; check_step $? indent.in tip_force_vs_disp.dat
echo "Pipeline terminado: $(date)"
