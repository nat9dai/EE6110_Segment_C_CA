# Deadlock recovery on the full main benchmark (Isaac Sim 5.1, n = 224)

**Result.** On the 224 main benchmark scenes, re-observing on deadlock (`safe_oacbf@recover=reobserve`,
"ours") keeps the occlusion-aware filter's low collision rate and removes most of its deadlocks. Against
`safe_oacbf` re-run in the same batch, deadlock fell from 59/224 to 8/224 and success rose from 64/224 to
89/224. Against the benchmark's unfiltered `ant_nom` (different batch, same scenes), ours collides far
less (3/224 against 17/224) but still succeeds less often (89/224 against 134/224). The deadlock rates of
ours (8/224) and `ant_nom` (3/224) differ within run-to-run noise.

Raw data `results/raw/dl_main/`, full tables
`results/summary_dl_main.{md,json}`, launcher `scripts/bench_dl_main.sh`, analysis
`scripts/analyze_dl_main.py`.

## Main metrics

Rates in %, Wilson 95% interval in brackets, n = 224 scenes, Isaac Sim 5.1. Ours comes from a separate
batch on the same 224 scenes (next section).

| method | success | collision | disturbed ≥1 | clean success | deadlock |
|---|---|---|---|---|---|
| `safe_oacbf` + re-observe (ours) | 40 [34–46] (89/224) | 1 [0–4] (3/224) | 16 [11–21] (35/224) | 34 [28–41] (77/224) | 4 [2–7] (8/224) |
| `safe_oacbf`, same batch as ours | 29 [23–35] (64/224) | 1 [0–3] (2/224) | 13 [10–18] (30/224) | 23 [18–29] (52/224) | 26 [21–32] (59/224) |

## Success by clutter level

Success in %, Isaac Sim 5.1.

| method | sparse (n = 60) | medium (n = 57) | dense (n = 48) | hidden (n = 59) |
|---|---|---|---|---|
| `safe_oacbf` + re-observe (ours) | 38 (23/60) | 35 (20/57) | 31 (15/48) | 53 (31/59) |
| `safe_oacbf`, same batch as ours | 33 (20/60) | 30 (17/57) | 23 (11/48) | 27 (16/59) |

## Ours against the other rows (Isaac Sim 5.1, n = 224 paired scene seeds)

`ours only` and `other only` count scenes where the event happened in that row only. p comes from an
exact paired test on those counts, corrected for the four comparisons per metric (McNemar, Holm).
Rows from the main batch are a different run of the same scene seeds: scenes settle a little
differently each run, and the target object matched in only 177 of 224 scenes.

| vs | batch | metric | ours | other | ours only | other only | p (corrected) |
|---|---|---|---|---|---|---|---|
| `safe_oacbf` | same | success | 89 | 64 | 33 | 8 | 0.0002 |
| `safe_oacbf` | same | collision | 3 | 2 | 1 | 0 | 1 |
| `safe_oacbf` | same | deadlock | 8 | 59 | 0 | 51 | 4e-15 |
| `ant_nom` | different | success | 89 | 134 | 21 | 66 | 6e-6 |
| `ant_nom` | different | collision | 3 | 17 | 2 | 16 | 0.005 |
| `ant_nom` | different | deadlock | 8 | 3 | 8 | 3 | 0.45 |
| `safe_oacbf` (main) | different | success | 89 | 71 | 45 | 27 | 0.044 |
| `safe_oacbf` (main) | different | collision | 3 | 2 | 3 | 2 | 1 |
| `safe_oacbf` (main) | different | deadlock | 8 | 59 | 5 | 56 | 2e-11 |
| `ant_cbf` | different | success | 89 | 126 | 22 | 59 | 0.0001 |
| `ant_cbf` | different | collision | 3 | 12 | 2 | 11 | 0.067 |
| `ant_cbf` | different | deadlock | 8 | 3 | 8 | 3 | 0.45 |

Clean success (success, no collision, nothing moved): ours 77/224, same-batch `safe_oacbf` 52/224
(corrected p = 0.0003), `ant_nom` 102/224 (p = 0.015), `ant_cbf` 97/224 (p = 0.029). Per clutter level
and for disturbed ≥1, see `results/summary_dl_main.md`.

## Setup

- **Platform.** Isaac Sim 5.1 with Isaac Lab 2.3.2.
- **Scenes.** Same launcher and flags as `scripts/bench_main.sh`: pool seeds 7, 8 and 9, 5 workers,
  20 scenes per kind per round, kinds sparse, medium, dense, hidden, main camera only. So scene seeds,
  object pools and depth-noise seeds match `results/raw/main`. RGB and traces not saved.
- **Arms.** `safe_oacbf` (the filter alone, the same-batch anchor) and `safe_oacbf@recover=reobserve`
  (ours), run one after the other on each scene from the same snapshot, images and depth noise.
- **Recovery.** As in `results/ab_deadlock.md`: a stall (1.5 s without 2 mm of progress) in transit or
  approach, farther than 15 mm from the grasp, re-runs perception with the main and both side cameras
  (images captured at the start of the trial), rebuilds both distance fields and resumes the same grasp
  candidate. At most 2 recoveries per trial.
- **Exclusions.** The 15 `main` scenes of `results/excluded_scenes.json`. This batch's own audit check
  flagged nothing. Scene 407009 had no valid target in the main batch, so it is dropped here too. That
  leaves the report's 224 scenes (60 sparse, 57 medium, 48 dense, 59 hidden), 448 trials in this batch.
- **Cost.** Recovery fired in 68 of 224 trials (0.40 events per trial). Mean wall time per trial
  12.6 s for ours against 12.0 s for the anchor.

## Notes

- **Run-to-run drift.** The anchor re-run here and the main batch's `safe_oacbf` agree on totals
  (success 64 against 71, deadlock 59 against 59, collision 2 against 2 of 224) but have the same
  per-scene outcome in only 142 of 224 scenes and the same target object in 177 of 224. So ours is
  compared to the main-batch rows only through totals and through noisier cross-batch pairs.
- **Noise inside this batch.** In the 156 scenes where ours fired no recovery, the two arms ran
  identical code. They still differ on success in 6 scenes, all in the anchor's favour. In those 6 the
  approach matched to the last digit and the difference arose in the close and lift (in 5 of them the
  fingers closed fully on nothing). That split would come
  out this one-sided about 3% of the time by chance. The anchor always ran first and ours second on
  each scene, so this may be a small trial-order effect. If it is, it works against ours, and the gain
  over the anchor is if anything understated. The main batch ran `safe_oacbf` last of 8 methods, so a
  swapped-order re-run would be the way to check whether order matters there too.
- **Where recovery helps.** Deadlock dropped on every clutter level (same batch, Isaac Sim 5.1): sparse
  12/60 to 1/60, medium 12/57 to 1/57, dense 13/48 to 5/48, hidden 22/59 to 1/59. Success rose clearly
  only on hidden scenes (16/59 to 31/59). On sparse, medium and dense scenes many rescued approaches end
  as slips, which is the benchmark's dominant failure for every method.

## Reproduce

```bash
scripts/bench_dl_main.sh 7 8 9
```

```bash
python3 scripts/analyze_dl_main.py
```

The first command runs 5 Isaac Sim 5.1 workers, about 12 minutes per pool seed on an RTX 5080, and
skips finished trials on a rerun. Do not run it while another Isaac process is running.
