#!/bin/bash
# run_stress.sh — calcula estrés por átomo para todas las configuraciones AM e IEX1
# Corre en pcHAL con lmp (serial o MPI).
#
# Uso: bash run_stress.sh [N_MPI]  (default: 4)
#
# Salida: COMPARISON/STRESS/dumps/{tag}_stress.dat

set -e

NPROC=${1:-4}
LMP="/home/alfredo/.local/bin/lmp"
HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="$HERE/../.."
STRESS_IN="$HERE/stress.in"
OUTDIR="$HERE/dumps"
mkdir -p "$OUTDIR"

POTFILE_CAO="$BASE/75SiO2_15-xNa2O_xK2O_10CaO/IPs/PMMCS/PMMCS_pedone_rc8p0.mod"
POTFILE_NOCA="$BASE/75SiO2_25-xNa2O_xK2O/IPs/PMMCS/PMMCS_pedone_rc8p0.mod"

CAO_AM_DIR="$BASE/75SiO2_15-xNa2O_xK2O_10CaO/STAGE1_MELTQUENCH/data/asmelted_723K"
CAO_IEX1_DIR="$BASE/75SiO2_15-xNa2O_xK2O_10CaO/STAGE3_IEX_PROTO1/data/iox1_723K"
NOCA_AM_DIR="$BASE/75SiO2_25-xNa2O_xK2O/STAGE1_MELTQUENCH/data/asmelted_723K"
NOCA_IEX1_DIR="$BASE/75SiO2_25-xNa2O_xK2O/STAGE3_IEX_PROTO1/data/iox1_723K"

# (xna_cao xka_cao xt_cao xna_noca xka_noca xt_noca)
declare -A XX_CONFIG
XX_CONFIG[20]="3 3 3 5 5 5"
XX_CONFIG[40]="6 6 6 10 10 10"
XX_CONFIG[60]="9 9 9 15 15 15"
XX_CONFIG[80]="12 12 12 20 20 20"
XX_CONFIG[100]="0 15 15 0 25 25"

run_lmp() {
    local datafile="$1" outbase="$2" potfile="$3"
    if [ ! -f "$datafile" ]; then
        echo "  AVISO: no existe $datafile — saltando"
        return
    fi
    if [ -f "${outbase}_stress.dat" ]; then
        echo "  Ya existe: $(basename ${outbase}_stress.dat) — saltando"
        return
    fi
    if [ "$NPROC" -gt 1 ]; then
        mpirun -np "$NPROC" "$LMP" -in "$STRESS_IN" \
            -var datafile "$datafile" -var outbase "$outbase" -var potfile "$potfile" \
            -log /dev/null -screen /dev/null
    else
        "$LMP" -in "$STRESS_IN" \
            -var datafile "$datafile" -var outbase "$outbase" -var potfile "$potfile" \
            -log /dev/null -screen /dev/null
    fi
    echo "  OK: $(basename ${outbase}_stress.dat)"
}

for XX in 20 40 60 80 100; do
    read xna_cao xka_cao xt_cao xna_noca xka_noca xt_noca <<< "${XX_CONFIG[$XX]}"
    echo "── XX=${XX}% ──"

    for r in 1 2 3; do
        # CaO — AM (Na)
        if [ "$xna_cao" -gt 0 ]; then
            run_lmp "$CAO_AM_DIR/AsMelted_723K_x${xna_cao}_r${r}_PMMCS_rc8p0.data" \
                    "$OUTDIR/cao_am_x${xna_cao}_r${r}" "$POTFILE_CAO"
        fi
        # CaO — AM (K) — solo si es diferente del de Na
        if [ "$xka_cao" -gt 0 ] && [ "$xka_cao" != "$xna_cao" ]; then
            run_lmp "$CAO_AM_DIR/AsMelted_723K_x${xka_cao}_r${r}_PMMCS_rc8p0.data" \
                    "$OUTDIR/cao_am_x${xka_cao}_r${r}" "$POTFILE_CAO"
        fi
        # CaO — IEX1
        run_lmp "$CAO_IEX1_DIR/IOX1_723K_xt${xt_cao}_r${r}_PMMCS_rc8p0.data" \
                "$OUTDIR/cao_iex1_xt${xt_cao}_r${r}" "$POTFILE_CAO"

        # Ca-free — AM (Na)
        if [ "$xna_noca" -gt 0 ]; then
            run_lmp "$NOCA_AM_DIR/AsMelted_723K_x${xna_noca}_r${r}_PMMCS_rc8p0.data" \
                    "$OUTDIR/noca_am_x${xna_noca}_r${r}" "$POTFILE_NOCA"
        fi
        # Ca-free — AM (K) — solo si es diferente del de Na
        if [ "$xka_noca" -gt 0 ] && [ "$xka_noca" != "$xna_noca" ]; then
            run_lmp "$NOCA_AM_DIR/AsMelted_723K_x${xka_noca}_r${r}_PMMCS_rc8p0.data" \
                    "$OUTDIR/noca_am_x${xka_noca}_r${r}" "$POTFILE_NOCA"
        fi
        # Ca-free — IEX1
        run_lmp "$NOCA_IEX1_DIR/IOX1_723K_xt${xt_noca}_r${r}_PMMCS_rc8p0.data" \
                "$OUTDIR/noca_iex1_xt${xt_noca}_r${r}" "$POTFILE_NOCA"
    done
done

echo ""
echo "Listo. Dumps en: $OUTDIR"
