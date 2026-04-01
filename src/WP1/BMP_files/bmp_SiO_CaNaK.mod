# Bertani et al. (2022). JACS 
# "include BMP_files/bmp_SiO_CaNaK.mod" in input.lammps
# Choose the type of atoms

################################################################################
###--- DEFINE GROUPS FOR SPECIES ---###
group Si type 1
group O  type 2
group Ca type 3
group Na type 4
group K  type 5

# Masses
mass    1 28.086    # Si
mass    2 15.9994   # O
mass    3 40.078    # Ca
mass    4 22.99     # Na
mass    5 39.100    # K

# Charges (Pedone/PMMCS effective charges)
set     type 1 charge  2.4000   # Si
set     type 2 charge -1.2000   # O
set     type 3 charge  1.2000   # Ca
set     type 4 charge  0.6000   # Na
set     type 5 charge  0.6000   # K

variable	rvdw equal 7.0
pair_style hybrid/overlay coul/dsf 0.2 12.0 table spline 10000 buck 7.0 nb3b/screened 
pair_coeff * * coul/dsf
#kspace_style pppm 1.0e-5


# pair_coeff    *    *           table "./Table_None.dat"         NONE  ${rvdw}
pair_coeff    1    2           table ../BMP_files/Table_PMMCS-Si-O.dat      albe  ${rvdw}
pair_coeff    2    2           table ../BMP_files/Table_PMMCS-O-O.dat       albe  ${rvdw}
pair_coeff    2    3           table ../BMP_files/Table_PMMCS-Ca-O.dat      albe  ${rvdw}
pair_coeff    2    4           table ../BMP_files/Table_PMMCS-Na-O.dat      albe  ${rvdw}
pair_coeff    2    5           table ../BMP_files/Table_BMP-K-O.dat         albe  ${rvdw}

# buckingham only for Si, Al
pair_coeff    1    1           buck  7.093669 0.975598 0.0

# tbp only for Si, with O 
pair_coeff * * nb3b/screened ../BMP_files/BMP-SiP.nb3b.shrm Si O NULL NULL NULL