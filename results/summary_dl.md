## Deadlock recovery A/B (Isaac Sim 5.1): 153 trials, 51 scenes x 3 arms, 253 distinct GSO objects

Scenes: dense and hidden, pool seeds 7 and 8, 15 per kind each (the controller-ablation scenes). Excluded by results/excluded_scenes.json: 8 [307003, 307009, 307010, 308001, 308004, 308005, 308008, 308013]. No valid target or no row: [407009]. Incomplete (dropped): [].

### All scenes

| arm | n | success | collision | disturbed ≥1 | clean success | deadlock | recoveries / trial | trials with a recovery | wall time / trial [s] |
|---|---|---|---|---|---|---|---|---|---|
| A0 reference | 51 | 24% [14–37] | 0% [0–7] | 20% [11–32] | 12% [6–23] | 24% [14–37] | 0.00 [0.00–0.00] | 0% [0–7] | 11.5 [10.1–13.1] |
| A1 re-observe on stall | 51 | 43% [31–57] | 0% [0–7] | 16% [8–28] | 29% [19–43] | 2% [0–10] | 0.37 [0.20–0.57] | 25% [16–39] | 11.8 [10.3–13.6] |
| A2 relax margin on stall | 51 | 24% [14–37] | 0% [0–7] | 20% [11–32] | 10% [4–21] | 24% [14–37] | 0.51 [0.27–0.75] | 25% [16–39] | 11.5 [10.2–12.7] |

### dense

| arm | n | success | collision | disturbed ≥1 | clean success | deadlock | recoveries / trial | trials with a recovery | wall time / trial [s] |
|---|---|---|---|---|---|---|---|---|---|
| A0 reference | 22 | 9% [3–28] | 0% [0–15] | 18% [7–39] | 9% [3–28] | 9% [3–28] | 0.00 [0.00–0.00] | 0% [0–15] | 12.4 [9.4–16.0] |
| A1 re-observe on stall | 22 | 18% [7–39] | 0% [0–15] | 5% [1–22] | 18% [7–39] | 0% [0–15] | 0.14 [0.00–0.27] | 14% [5–33] | 12.3 [9.4–15.5] |
| A2 relax margin on stall | 22 | 9% [3–28] | 0% [0–15] | 9% [3–28] | 9% [3–28] | 9% [3–28] | 0.27 [0.00–0.55] | 14% [5–33] | 11.0 [8.7–13.2] |

### hidden

| arm | n | success | collision | disturbed ≥1 | clean success | deadlock | recoveries / trial | trials with a recovery | wall time / trial [s] |
|---|---|---|---|---|---|---|---|---|---|
| A0 reference | 29 | 34% [20–53] | 0% [0–12] | 21% [10–38] | 14% [5–31] | 34% [20–53] | 0.00 [0.00–0.00] | 0% [0–12] | 10.8 [9.6–12.1] |
| A1 re-observe on stall | 29 | 62% [44–77] | 0% [0–12] | 24% [12–42] | 38% [23–56] | 3% [1–17] | 0.55 [0.28–0.86] | 34% [20–53] | 11.5 [10.0–13.2] |
| A2 relax margin on stall | 29 | 34% [20–53] | 0% [0–12] | 28% [15–46] | 10% [4–26] | 34% [20–53] | 0.69 [0.34–1.03] | 34% [20–53] | 11.8 [10.3–13.4] |

### Paired McNemar vs A0 (exact, two-sided)

Holm over the two comparisons, per metric (all scenes). `A0 only` / `arm only` = discordant scenes.

| scenes | metric | arm | n | A0 | arm | A0 only | arm only | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| all | success | A1 | 51 | 24% | 43% | 0 | 10 | 0.002 | 0.0039 |
| all | success | A2 | 51 | 24% | 24% | 0 | 0 | 1 | 1 |
| all | deadlock | A1 | 51 | 24% | 2% | 11 | 0 | 0.00098 | 0.002 |
| all | deadlock | A2 | 51 | 24% | 24% | 0 | 0 | 1 | 1 |
| all | collision | A1 | 51 | 0% | 0% | 0 | 0 | 1 | 1 |
| all | collision | A2 | 51 | 0% | 0% | 0 | 0 | 1 | 1 |
| all | clean_success | A1 | 51 | 12% | 29% | 1 | 10 | 0.012 | 0.023 |
| all | clean_success | A2 | 51 | 12% | 10% | 1 | 0 | 1 | 1 |
| dense | success | A1 | 22 | 9% | 18% | 0 | 2 | 0.5 | – |
| dense | success | A2 | 22 | 9% | 9% | 0 | 0 | 1 | – |
| dense | deadlock | A1 | 22 | 9% | 0% | 2 | 0 | 0.5 | – |
| dense | deadlock | A2 | 22 | 9% | 9% | 0 | 0 | 1 | – |
| dense | collision | A1 | 22 | 0% | 0% | 0 | 0 | 1 | – |
| dense | collision | A2 | 22 | 0% | 0% | 0 | 0 | 1 | – |
| dense | clean_success | A1 | 22 | 9% | 18% | 0 | 2 | 0.5 | – |
| dense | clean_success | A2 | 22 | 9% | 9% | 0 | 0 | 1 | – |
| hidden | success | A1 | 29 | 34% | 62% | 0 | 8 | 0.0078 | – |
| hidden | success | A2 | 29 | 34% | 34% | 0 | 0 | 1 | – |
| hidden | deadlock | A1 | 29 | 34% | 3% | 9 | 0 | 0.0039 | – |
| hidden | deadlock | A2 | 29 | 34% | 34% | 0 | 0 | 1 | – |
| hidden | collision | A1 | 29 | 0% | 0% | 0 | 0 | 1 | – |
| hidden | collision | A2 | 29 | 0% | 0% | 0 | 0 | 1 | – |
| hidden | clean_success | A1 | 29 | 14% | 38% | 1 | 8 | 0.039 | – |
| hidden | clean_success | A2 | 29 | 14% | 10% | 1 | 0 | 1 | – |

### Recovery events (all scenes)

| arm | events | in transit | in approach | effective (changed the barriers) | first event resumed to a full approach | outcomes (success / slip / deadlock / no grasp) |
|---|---|---|---|---|---|---|
| A0 reference | 0 | 0 | 0 | 0 | 0 of 0 | 12 / 23 / 12 / 4 |
| A1 re-observe on stall | 19 | 0 | 19 | 13 | 6 of 13 | 22 / 24 / 1 / 4 |
| A2 relax margin on stall | 26 | 0 | 26 | 13 | 0 of 13 | 12 / 23 / 12 / 4 |
