#!/bin/bash
# run_parallel_mpi_24cores.sh

date


################################################################################
###--- PREPARE THE AS-MELTED GLASS FOR DIFFERENT Na/K RATIOS ---###


cd MELTQUENCH

t_final=(300)


# --- Define x array and corresponding experimental densities for melt-quench simulations
# --- x_values and dens_values are parsed from the file "MeltQuench.in"
x_values=(0 1 2 3 4 5 6 9 12 15)
dens_values=(2.48211 2.46275 2.46118 2.481 2.45889 2.46098 2.48284 2.47479 2.47525 2.45893)

for ((i=0; i<${#x_values[@]}; i++)); do
  x=${x_values[i]}
  dens=${dens_values[i]}

  mpirun -np 48 lmp -in MeltQuench.in -var target_density ${dens} -var val_x ${x} -var T_end ${t_final}
  echo "  - MeltQuench.in simulation: x=${x}, density=${dens} g/cm³, t_final=${t_final}"
done

################################################################################
###--- INTERATOMIC POTENTIAL ---###

cd ../MSD

for ((i=0; i<${#x_values[@]}; i++)); do
  x=${x_values[i]}
  
  mpirun -np 48 lmp -in MSD_Na-K.in -var val_x ${x} -var T_end ${t_final}
  echo "  - MSD_Na-K.in simulation: x=${x}, t_final=${t_final}"
done


################################################################################
###--- INTERATOMIC POTENTIAL ---###

cd ../YOUNG_MODULUS_STRAIN_STRESS

for ((i=0; i<${#x_values[@]}; i++)); do
  x=${x_values[i]}
  mpirun -np 48 lmp -in input_lammps_x.in -var val_x ${x} -var T_end ${t_final}
  echo "  - input_lammps_x.in simulation: x=${x}, t_final=${t_final}"
  mpirun -np 48 lmp -in input_lammps_y.in -var val_x ${x} -var T_end ${t_final}
  echo "  - input_lammps_y.in simulation: x=${x}, t_final=${t_final}"
  mpirun -np 48 lmp -in input_lammps_z.in -var val_x ${x} -var T_end ${t_final}
  echo "  - input_lammps_z.in simulation: x=${x}, t_final=${t_final}"
done
