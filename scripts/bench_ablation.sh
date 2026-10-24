#!/bin/bash
# Ablations on dense + hidden scenes (2 object-pool rounds x 15 scenes per kind).
cd "$(dirname "$0")/.."
M="safe_oacbf,safe_oacbf@views=main+left+right,ant_nom@views=main+left+right,ant_cbf@views=main+left+right"
M="$M,safe_oacbf@alpha=1,safe_oacbf@alpha=2,safe_oacbf@alpha=15"
M="$M,safe_oacbf@margin_unk=0.0,safe_oacbf@margin_unk=0.015,safe_oacbf@margin_obs=0.005,safe_oacbf@margin_obs=0.02"
M="$M,safe_oacbf@carve_radius=0.0,safe_oacbf@carve_radius=0.04,safe_oacbf@unk_soft=false"
M="$M,safe_oacbf@noise=0,safe_oacbf@noise=3,ant_nom@noise=0,ant_nom@noise=3"
for PS in 7 8; do
  python3 scripts/launch_benchmark.py --workers 5 --stagger 6 -- --exp abl --pool-seed $PS --n-scenes 15 --threads 2 \
    --kinds dense,hidden --methods "$M" --save-rgb 0
done
echo ABL_DONE
