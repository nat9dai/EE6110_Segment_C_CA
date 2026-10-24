#!/bin/bash
# Main benchmark: 3 object-pool rounds x 5 workers, 8 methods, main camera only.
# Usage: scripts/bench_main.sh [scenes_per_kind_per_round]
cd "$(dirname "$0")/.."
N=${1:-20}
for PS in 7 8 9; do
  python3 scripts/launch_benchmark.py --workers 5 --stagger 6 -- --exp main --pool-seed $PS --n-scenes $N --threads 2 \
    --kinds sparse,medium,dense,hidden --methods td_nom,ant_nom,ant_apf,ant_cbf,ant_oacbf,safe_nom,safe_cbf,safe_oacbf \
    --save-rgb 2 --trace safe_oacbf,ant_nom
done
echo ALL_ROUNDS_DONE
