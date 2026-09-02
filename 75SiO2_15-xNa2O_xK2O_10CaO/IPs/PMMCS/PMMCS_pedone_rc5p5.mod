# PMMCS potential — pair_pedone native implementation, rc_pedone = 5.5 A
# Pedone et al., J. Phys. Chem. B 110, 11780 (2006), Table 2
# U(r) = D*(1 - exp(-a*(r-r0)))^2 - D + C/r^12  (short-range, no Coulomb)
# Coulomb via coul/long + pppm.
#
# Atom types: Si=1  O=2  Ca=3  Na=4  K=5
# Charges:    +2.4 -1.2  +1.2  +0.6 +0.6

mass 1 28.086    # Si
mass 2 15.9994   # O
mass 3 40.078    # Ca
mass 4 22.990    # Na
mass 5 39.100    # K

set type 1 charge  2.4
set type 2 charge -1.2
set type 3 charge  1.2
set type 4 charge  0.6
set type 5 charge  0.6

pair_style hybrid/overlay pedone 5.5 coul/long 12.0
kspace_style pppm 1.0e-4

# All pairs — Coulomb
pair_coeff * * coul/long

# Short-range Morse + C/r^12 (M-O and O-O only)
# pair_coeff I J pedone D(eV) a(1/A) r0(A) C(eV.A^12)
pair_coeff 1 2 pedone 0.340554 2.006700 2.100000  1.0   # Si-O
pair_coeff 2 2 pedone 0.042395 1.379316 3.618701 22.0   # O-O
pair_coeff 2 3 pedone 0.030211 2.241334 2.923245  5.0   # O-Ca
pair_coeff 2 4 pedone 0.023363 1.763867 3.006315  5.0   # O-Na
pair_coeff 2 5 pedone 0.011612 2.062605 3.305308  5.0   # O-K
