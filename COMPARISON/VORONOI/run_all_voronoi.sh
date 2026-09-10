#!/usr/bin/env bash
# run_all_voronoi.sh — lanza LAMMPS voronoi.in para todos los .data necesarios.
# Salida: VORONOI/{tag}_vol.dat  y  VORONOI/{tag}_neigh.dat

set -euo pipefail

LMP=/home/alfredo/Software/lammps-29Aug2024/build/lmp
INP=$(dirname "$0")/voronoi.in
OUTDIR=$(dirname "$0")

BASE=/home/alfredo/Simulations_MD_LAMMPS/IEX_SLS

CAO_AM=$BASE/75SiO2_15-xNa2O_xK2O_10CaO/STAGE1_MELTQUENCH/data/asmelted_723K
CAO_IEX=$BASE/75SiO2_15-xNa2O_xK2O_10CaO/STAGE3_IEX_PROTO1/data/iox1_723K
NOCA_AM=$BASE/75SiO2_25-xNa2O_xK2O/STAGE1_MELTQUENCH/data/asmelted_723K
NOCA_IEX=$BASE/75SiO2_25-xNa2O_xK2O/STAGE3_IEX_PROTO1/data/iox1_723K

run_lmp() {
    local datafile=$1
    local tag=$2
    local vol=${OUTDIR}/${tag}_vol.dat
    if [[ -f "$vol" ]]; then
        echo "  skip (existe): $tag"
        return
    fi
    echo "  $tag"
    $LMP -in "$INP" -var datafile "$datafile" -var outbase "${OUTDIR}/${tag}" \
         -log none -screen none
}

echo "=== CaO — AM ==="
for X in 0 3 6 9 12 15; do
    for R in 1 2 3; do
        run_lmp "$CAO_AM/AsMelted_723K_x${X}_r${R}_PMMCS_rc8p0.data" \
                "cao_am_x${X}_r${R}"
    done
done

echo "=== CaO — IEX1 ==="
for X in 3 6 9 12 15; do
    for R in 1 2 3; do
        run_lmp "$CAO_IEX/IOX1_723K_xt${X}_r${R}_PMMCS_rc8p0.data" \
                "cao_iex1_xt${X}_r${R}"
    done
done

echo "=== Ca-free — AM ==="
for X in 0 5 10 15 20 25; do
    for R in 1 2 3; do
        run_lmp "$NOCA_AM/AsMelted_723K_x${X}_r${R}_PMMCS_rc8p0.data" \
                "noca_am_x${X}_r${R}"
    done
done

echo "=== Ca-free — IEX1 ==="
for X in 5 10 15 20 25; do
    for R in 1 2 3; do
        run_lmp "$NOCA_IEX/IOX1_723K_xt${X}_r${R}_PMMCS_rc8p0.data" \
                "noca_iex1_xt${X}_r${R}"
    done
done

echo "=== Listo ==="
