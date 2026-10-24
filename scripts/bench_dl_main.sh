#!/bin/bash
# Deadlock recovery on the main benchmark scenes: same launcher and flags as bench_main.sh (3 pool rounds
# x 5 workers, 20 scenes per kind, main camera only), two arms re-run in one batch so they pair per scene.
# Usage: scripts/bench_dl_main.sh [pool seeds, default "7 8 9"]
cd "$(dirname "$0")/.."
M="safe_oacbf,safe_oacbf@recover=reobserve"
for PS in ${@:-7 8 9}; do
  python3 scripts/launch_benchmark.py --workers 5 --stagger 6 -- --exp dl_main --pool-seed $PS --n-scenes 20 --threads 2 \
    --kinds sparse,medium,dense,hidden --methods "$M" --save-rgb 0
done
echo DL_MAIN_DONE
