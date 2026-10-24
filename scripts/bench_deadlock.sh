#!/bin/bash
# Deadlock-recovery A/B on the controller-ablation scenes (dense + hidden, 15 scenes per kind per round).
# 5 workers as in bench_ablation.sh: each worker's object pool depends on its index, so the worker
# count must match for the scenes to be identical.  Usage: scripts/bench_deadlock.sh [pool seeds, default "7 8"]
cd "$(dirname "$0")/.."
M="safe_oacbf,safe_oacbf@recover=reobserve,safe_oacbf@recover=relax"
for PS in ${@:-7 8}; do
  python3 scripts/launch_benchmark.py --workers 5 --stagger 6 -- --exp dl --pool-seed $PS --n-scenes 15 --threads 2 \
    --kinds dense,hidden --methods "$M" --trace "$M" --save-rgb 0
done
echo DL_DONE
