#!/usr/bin/env bash
#SBATCH --job-name=indent112_cao_x0_r2
#SBATCH --account=euhpc_b38_167
#SBATCH --partition=dcgp_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=112
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=alfredo.sanchez@tnuni.sk
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Solo STEP 4 (indent) con 112 cores; slab2x2_tip.data ya generado por el pipeline
module purge
module load profile/chem-phys
module load lammps/29aug2024
export OMP_NUM_THREADS=1
# FFTW de compilacion (evita segfault en fftw_destroy_plan)
export LD_LIBRARY_PATH=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-12.2.0/fftw-3.3.10-f3ms2lmqzjugb43xzrxjkvmvohumyruy/lib:$LD_LIBRARY_PATH
mkdir -p logs
cd "${SLURM_SUBMIT_DIR}"
[[ -s slab2x2_tip.data ]] || { echo "FALLO: falta slab2x2_tip.data"; exit 1; }
mpirun --bind-to none -np "${SLURM_NTASKS}" $HOME/local/bin/lmp -in indent.in -log log.4_indent
[[ -f tip_force_vs_disp.dat ]] || { echo "FALLO: indent.in no genero tip_force_vs_disp.dat"; exit 1; }
echo "Indent terminado: $(date)"
