# Deadlock recovery for the occlusion-aware CBF (paired A/B, Isaac Sim 5.1)

**Result.** Re-observing the scene with the two side cameras when the arm stalls removes almost all
deadlocks of `safe_oacbf` and raises grasp success. Deadlock fell from 12/51 to 1/51 and success rose
from 12/51 to 22/51 (51 paired dense and hidden scenes, Isaac Sim 5.1). Both changes hold after Holm
correction (p = 0.002 and p = 0.004). Collisions stayed at 0/51. Relaxing the unknown-space margin on
stall changed no success or deadlock outcome (p = 1).

Raw data in `results/raw/dl/`, summary in
`results/summary_dl.{md,json}`, figure `results/figures/dl_outcomes.png`.

## Setup

- **Platform.** Isaac Sim 5.1 with Isaac Lab 2.3.2.
- **Scenes.** The controller-ablation scene seeds: dense and hidden, object-pool seeds 7 and 8, 15 scenes
  per kind each, launched by `scripts/bench_deadlock.sh` with 5 workers. Each worker's object pool
  depends on its index, so 5 workers give the same seeds and pools as `scripts/bench_ablation.sh`.
- **Exclusions.** All lists in `results/excluded_scenes.json` apply. 8 of the 60 scenes are on them
  (307003, 307009, 307010, 308001, 308004, 308005, 308008, 308013) and scene 407009 has no valid
  target. That leaves the same 51 scenes as the controller ablation: 22 dense and 29 hidden, 153 trials.
- **Pairing.** In each scene the three arms start from the same settled snapshot, use the same camera
  images and the same depth noise, in the same worker process.
- **Metrics.** As in the benchmark. Success: the target is held at least 8 cm up after the hold.
  Deadlock: the trial fails and its last grasp attempt ended in a stall (1.5 s without 2 mm of progress)
  or a timeout. Clean success: success with no collision and no other object moved.
- **Trigger.** A stall in transit or approach while the `oacbf` executor runs. An approach that stalls
  within 15 mm of the grasp is not a trigger, because the original code closes the gripper there anyway.
  At most 2 recoveries per trial. After that a stall falls back to the original behaviour (back off, try
  the next grasp candidate, finally fail as deadlock).
- **Arms.**
  - A0 `safe_oacbf`: unchanged reference.
  - A1 `safe_oacbf@recover=reobserve`: re-run perception with the main, left and right cameras, rebuild
    the obstacle and unknown-space distance fields and barriers, resume the same grasp candidate from
    the current joint state.
  - A2 `safe_oacbf@recover=relax`: set the unknown-space margin from 5 mm to 0 for the rest of the trial
    (the barrier stays soft), resume.
- **Resume.** Transit restarts toward the same pre-grasp. Approach restarts its moving reference at the
  gripper's current depth along the approach axis, so it does not jump back to the pre-grasp.
- **Statistics.** Wilson 95% intervals for rates, bootstrap 95% intervals for means, exact two-sided
  McNemar test against A0 on the 51 paired scenes, Holm correction over the two comparisons (A1, A2)
  for each metric. Per-kind p-values are not adjusted.

## Results (Isaac Sim 5.1)

Counts with Wilson 95% intervals. Recoveries and wall time: mean with bootstrap 95% interval.

### All scenes (n = 51 per arm)

| arm | success | collision | disturbed ≥1 | clean success | deadlock | recoveries per trial | trials with a recovery | wall time per trial [s] |
|---|---|---|---|---|---|---|---|---|
| A0 `safe_oacbf` (reference) | 12/51 (24% [14, 37]) | 0/51 (0% [0, 7]) | 10/51 (20% [11, 32]) | 6/51 (12% [6, 23]) | 12/51 (24% [14, 37]) | 0.00 | 0/51 | 11.5 [10.1, 13.1] |
| A1 `safe_oacbf@recover=reobserve` | 22/51 (43% [31, 57]) | 0/51 (0% [0, 7]) | 8/51 (16% [8, 28]) | 15/51 (29% [19, 43]) | 1/51 (2% [0, 10]) | 0.37 [0.20, 0.57] | 13/51 (25% [16, 39]) | 11.8 [10.3, 13.6] |
| A2 `safe_oacbf@recover=relax` | 12/51 (24% [14, 37]) | 0/51 (0% [0, 7]) | 10/51 (20% [11, 32]) | 5/51 (10% [4, 21]) | 12/51 (24% [14, 37]) | 0.51 [0.27, 0.75] | 13/51 (25% [16, 39]) | 11.5 [10.2, 12.7] |

### Dense (n = 22 per arm)

| arm | success | collision | disturbed ≥1 | clean success | deadlock | recoveries per trial | trials with a recovery | wall time per trial [s] |
|---|---|---|---|---|---|---|---|---|
| A0 `safe_oacbf` (reference) | 2/22 (9% [3, 28]) | 0/22 (0% [0, 15]) | 4/22 (18% [7, 39]) | 2/22 (9% [3, 28]) | 2/22 (9% [3, 28]) | 0.00 | 0/22 | 12.4 [9.4, 16.0] |
| A1 `safe_oacbf@recover=reobserve` | 4/22 (18% [7, 39]) | 0/22 (0% [0, 15]) | 1/22 (5% [1, 22]) | 4/22 (18% [7, 39]) | 0/22 (0% [0, 15]) | 0.14 [0.00, 0.27] | 3/22 (14% [5, 33]) | 12.3 [9.4, 15.5] |
| A2 `safe_oacbf@recover=relax` | 2/22 (9% [3, 28]) | 0/22 (0% [0, 15]) | 2/22 (9% [3, 28]) | 2/22 (9% [3, 28]) | 2/22 (9% [3, 28]) | 0.27 [0.00, 0.55] | 3/22 (14% [5, 33]) | 11.0 [8.7, 13.2] |

### Hidden (n = 29 per arm)

| arm | success | collision | disturbed ≥1 | clean success | deadlock | recoveries per trial | trials with a recovery | wall time per trial [s] |
|---|---|---|---|---|---|---|---|---|
| A0 `safe_oacbf` (reference) | 10/29 (34% [20, 53]) | 0/29 (0% [0, 12]) | 6/29 (21% [10, 38]) | 4/29 (14% [5, 31]) | 10/29 (34% [20, 53]) | 0.00 | 0/29 | 10.8 [9.6, 12.1] |
| A1 `safe_oacbf@recover=reobserve` | 18/29 (62% [44, 77]) | 0/29 (0% [0, 12]) | 7/29 (24% [12, 42]) | 11/29 (38% [23, 56]) | 1/29 (3% [1, 17]) | 0.55 [0.28, 0.86] | 10/29 (34% [20, 53]) | 11.5 [10.0, 13.2] |
| A2 `safe_oacbf@recover=relax` | 10/29 (34% [20, 53]) | 0/29 (0% [0, 12]) | 8/29 (28% [15, 46]) | 3/29 (10% [4, 26]) | 10/29 (34% [20, 53]) | 0.69 [0.34, 1.03] | 10/29 (34% [20, 53]) | 11.8 [10.3, 13.4] |

### Paired McNemar against A0

"A0 only" and "arm only" count the discordant scenes (the event happened in that arm only).

| scenes | metric | comparison | A0 only | arm only | p | p (Holm) |
|---|---|---|---|---|---|---|
| all (n = 51) | deadlock | A1 vs A0 | 11 | 0 | 0.00098 | **0.002** |
| all (n = 51) | deadlock | A2 vs A0 | 0 | 0 | 1 | 1 |
| all (n = 51) | success | A1 vs A0 | 0 | 10 | 0.002 | **0.0039** |
| all (n = 51) | success | A2 vs A0 | 0 | 0 | 1 | 1 |
| all (n = 51) | clean success | A1 vs A0 | 1 | 10 | 0.012 | 0.023 |
| all (n = 51) | clean success | A2 vs A0 | 1 | 0 | 1 | 1 |
| all (n = 51) | collision | A1, A2 vs A0 | 0 | 0 | 1 | 1 |
| hidden (n = 29) | deadlock | A1 vs A0 | 9 | 0 | 0.0039 | not adjusted |
| hidden (n = 29) | success | A1 vs A0 | 0 | 8 | 0.0078 | not adjusted |
| dense (n = 22) | deadlock | A1 vs A0 | 2 | 0 | 0.5 | not adjusted |
| dense (n = 22) | success | A1 vs A0 | 0 | 2 | 0.5 | not adjusted |

A stricter Holm correction over all four primary tests (deadlock and success, A1 and A2) gives
p = 0.0039 for deadlock and p = 0.0059 for success for A1. The conclusion does not change.

## How the rescue happens (Isaac Sim 5.1, n = 51)

- **Outcome changes.** A0 to A1: 10 deadlocks became successes, 1 became a slip, 1 stayed a deadlock.
  No other outcome changed (12 successes, 23 slips and 4 no-grasp trials are the same in both arms).
- **Same candidate or next candidate.** In 6 of the 10 rescued scenes the same grasp finished its
  approach right after re-observing. In the other 4 the first grasp stalled again and the next
  candidate succeeded with the new map. In 3 of the 6 trials that stalled again, the arm stopped at an
  obstacle that the side views had added to the map.
- **Barrier size.** At the first re-observation the unknown-space barrier shrank from a median of
  7,118 to 805 voxels (median 13% kept, 13 events). Re-running perception took 0.56 s on average
  (at most 1.3 s).
- **Relax.** A2 triggered in the same 13 trials. In all 13 the arm stalled again after the first
  event and again after the second, so every A2 trial ended as in A0. Dropping the 5 mm margin lets
  the arm creep a few millimetres before it meets the unknown region itself.
- **Where.** All 45 recovery events (19 in A1, 26 in A2) happened in approach. No transit stall occurred.

## Notes and limits

- **Side images come from the start of the trial.** The side cameras are static and their images were
  captured once per scene with the arm at home. A real robot that re-captures mid-trial would also see
  its own arm and any object it has already pushed. A1 therefore shows the value of the extra views,
  not of a new capture.
- **The second recovery is a no-op.** Re-running perception on the same three views, or setting an
  already zero margin to zero, changes nothing. The second event only gives the same candidate one more
  1.5 s window, and in this run it never resolved a stall (19 second events, 6 in A1 and 13 in A2,
  none finished the approach).
- **Comparison with side cameras from the start.** In the controller ablation (a different run of the
  same 51 scenes, Isaac Sim 5.1), `safe_oacbf@views=main+left+right` reached 28/51 success and 1/51
  deadlock. Re-observing on demand reaches the same deadlock rate but less of the success gain
  (22/51), likely because the grasp is still chosen from the main view alone. Only the totals are
  comparable here, not single scenes (next point).
- **Run-to-run variation.** Scenes settle a little differently in each run. Re-running A0 gave the same
  success total as the ablation's `safe_oacbf` (12/51 in both) but the same per-scene outcome in only
  27 of 51 scenes, and 12 against 17 deadlocks. The main benchmark and the ablation also agree on only
  27 of 51 scenes and pick a different target object in 13 of 51. So paired comparisons are valid only
  within one run, which is how this A/B is built.
- **Noise floor inside this run.** In the 38 scenes with no recovery event, all three arms had the same
  success, collision and failure type. Only the count of disturbed objects differed (3 of 38 scenes for
  A1, 4 of 38 for A2), from physics noise. The small differences in disturbed ≥1 and in clean success
  for A2 are within this noise.

## Reproduce

```bash
scripts/bench_deadlock.sh 7 8
```

```bash
python3 scripts/analyze_dl.py
```

The first command runs 5 Isaac Sim 5.1 workers (about 6 minutes per pool seed on an RTX 5080) and appends
to `results/raw/dl/w0*.jsonl`. Finished trials are skipped on a rerun. Do not run it while another Isaac
process is running. Unit tests: `~/isaacsim/python.sh -m pytest -q tests/test_recovery.py`.
