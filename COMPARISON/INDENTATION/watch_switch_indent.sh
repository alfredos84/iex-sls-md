#!/usr/bin/env bash
# uso: watch_switch_indent.sh <carpeta_caso> <jobid_pipeline>
D=$1; J=$2
cd "$D" || exit 1
until grep -q "Total wall time" log.3_tip 2>/dev/null && [ -s slab2x2_tip.data ]; do sleep 60; done
sleep 10
echo "$(date) relax+tip listos; cancelando $J y lanzando indent 112c"
scancel "$J"
while squeue -h -j "$J" 2>/dev/null | grep -q .; do sleep 5; done
rm -f tip_force_vs_disp.dat indent.lammpstrj restart.indent.a restart.indent.b
sbatch submit_indent_leonardo.slurm
