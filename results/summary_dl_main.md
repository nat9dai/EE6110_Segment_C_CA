## Deadlock recovery on the main benchmark scenes (Isaac Sim 5.1): 448 trials, 224 scenes x 2 arms

Same launcher and flags as scripts/bench_main.sh (pool seeds 7, 8, 9; 5 workers; 20 scenes per kind; main camera only). Excluded: the 15 'main' scenes of results/excluded_scenes.json, plus this batch's own audit flags []. Dropped: incomplete in this batch [], not in the main rows [407009], main scenes missing here [].

Rows marked `main:` are the main benchmark's own trials on the same scene seeds, from a different batch.

### All scenes

| method | n | success | collision | disturbed ≥1 | clean success | deadlock | recoveries / trial | trials with a recovery |
|---|---|---|---|---|---|---|---|---|
| safe_oacbf@recover=reobserve | 224 | 40% [34–46] | 1% [0–4] | 16% [11–21] | 34% [28–41] | 4% [2–7] | 0.40 [0.32–0.49] | 30% [25–37] |
| safe_oacbf | 224 | 29% [23–35] | 1% [0–3] | 13% [10–18] | 23% [18–29] | 26% [21–32] | 0.00 [0.00–0.00] | 0% [0–2] |
| main:safe_oacbf | 224 | 32% [26–38] | 1% [0–3] | 17% [13–23] | 22% [17–28] | 26% [21–32] | 0.00 [0.00–0.00] | 0% [0–2] |
| main:td_nom | 224 | 47% [41–54] | 16% [12–21] | 21% [16–27] | 33% [27–39] | 22% [17–28] | 0.00 [0.00–0.00] | 0% [0–2] |
| main:ant_nom | 224 | 60% [53–66] | 8% [5–12] | 22% [17–28] | 46% [39–52] | 1% [0–4] | 0.00 [0.00–0.00] | 0% [0–2] |
| main:ant_apf | 224 | 34% [28–41] | 3% [2–6] | 10% [7–15] | 29% [23–35] | 38% [31–44] | 0.00 [0.00–0.00] | 0% [0–2] |
| main:ant_cbf | 224 | 56% [50–63] | 5% [3–9] | 19% [15–25] | 43% [37–50] | 1% [0–4] | 0.00 [0.00–0.00] | 0% [0–2] |
| main:ant_oacbf | 224 | 25% [20–32] | 2% [1–5] | 13% [9–18] | 19% [15–25] | 44% [37–50] | 0.00 [0.00–0.00] | 0% [0–2] |
| main:safe_nom | 224 | 50% [44–57] | 4% [2–7] | 21% [16–27] | 38% [32–44] | 0% [0–2] | 0.00 [0.00–0.00] | 0% [0–2] |
| main:safe_cbf | 224 | 48% [42–55] | 3% [1–6] | 21% [16–26] | 36% [30–42] | 0% [0–2] | 0.00 [0.00–0.00] | 0% [0–2] |

### Success by clutter level (%)

| method | sparse (n = 60) | medium (n = 57) | dense (n = 48) | hidden (n = 59) |
|---|---|---|---|---|
| safe_oacbf@recover=reobserve | 38 | 35 | 31 | 53 |
| safe_oacbf | 33 | 30 | 23 | 27 |
| main:safe_oacbf | 37 | 32 | 29 | 29 |
| main:td_nom | 53 | 42 | 42 | 51 |
| main:ant_nom | 62 | 51 | 42 | 81 |
| main:ant_apf | 45 | 21 | 8 | 58 |
| main:ant_cbf | 62 | 42 | 40 | 78 |
| main:ant_oacbf | 35 | 32 | 25 | 10 |
| main:safe_nom | 53 | 47 | 38 | 61 |
| main:safe_cbf | 52 | 42 | 33 | 63 |

### sparse

| method | n | success | collision | disturbed ≥1 | clean success | deadlock | recoveries / trial |
|---|---|---|---|---|---|---|---|
| safe_oacbf@recover=reobserve | 60 | 38% [27–51] | 2% [0–9] | 5% [2–14] | 35% [24–48] | 2% [0–9] | 0.23 [0.12–0.35] |
| safe_oacbf | 60 | 33% [23–46] | 2% [0–9] | 5% [2–14] | 30% [20–43] | 20% [12–32] | 0.00 [0.00–0.00] |
| main:safe_oacbf | 60 | 37% [26–49] | 2% [0–9] | 7% [3–16] | 33% [23–46] | 22% [13–34] | 0.00 [0.00–0.00] |
| main:ant_nom | 60 | 62% [49–73] | 5% [2–14] | 12% [6–22] | 53% [41–65] | 2% [0–9] | 0.00 [0.00–0.00] |
| main:ant_cbf | 60 | 62% [49–73] | 3% [1–11] | 12% [6–22] | 52% [39–64] | 2% [0–9] | 0.00 [0.00–0.00] |

### medium

| method | n | success | collision | disturbed ≥1 | clean success | deadlock | recoveries / trial |
|---|---|---|---|---|---|---|---|
| safe_oacbf@recover=reobserve | 57 | 35% [24–48] | 0% [0–6] | 18% [10–29] | 32% [21–44] | 2% [0–9] | 0.32 [0.18–0.46] |
| safe_oacbf | 57 | 30% [20–43] | 0% [0–6] | 12% [6–23] | 26% [17–39] | 21% [12–33] | 0.00 [0.00–0.00] |
| main:safe_oacbf | 57 | 32% [21–44] | 0% [0–6] | 19% [11–31] | 25% [15–37] | 26% [17–39] | 0.00 [0.00–0.00] |
| main:ant_nom | 57 | 51% [38–63] | 4% [1–12] | 19% [11–31] | 40% [29–53] | 2% [0–9] | 0.00 [0.00–0.00] |
| main:ant_cbf | 57 | 42% [30–55] | 4% [1–12] | 16% [9–27] | 35% [24–48] | 0% [0–6] | 0.00 [0.00–0.00] |

### dense

| method | n | success | collision | disturbed ≥1 | clean success | deadlock | recoveries / trial |
|---|---|---|---|---|---|---|---|
| safe_oacbf@recover=reobserve | 48 | 31% [20–45] | 2% [0–11] | 29% [18–43] | 29% [18–43] | 10% [5–22] | 0.48 [0.29–0.69] |
| safe_oacbf | 48 | 23% [13–37] | 2% [0–11] | 23% [13–37] | 21% [12–34] | 27% [17–41] | 0.00 [0.00–0.00] |
| main:safe_oacbf | 48 | 29% [18–43] | 2% [0–11] | 23% [13–37] | 19% [10–32] | 19% [10–32] | 0.00 [0.00–0.00] |
| main:ant_nom | 48 | 42% [29–56] | 15% [7–27] | 42% [29–56] | 27% [17–41] | 2% [0–11] | 0.00 [0.00–0.00] |
| main:ant_cbf | 48 | 40% [27–54] | 8% [3–20] | 42% [29–56] | 21% [12–34] | 4% [1–14] | 0.00 [0.00–0.00] |

### hidden

| method | n | success | collision | disturbed ≥1 | clean success | deadlock | recoveries / trial |
|---|---|---|---|---|---|---|---|
| safe_oacbf@recover=reobserve | 59 | 53% [40–65] | 2% [0–9] | 14% [7–25] | 41% [29–53] | 2% [0–9] | 0.59 [0.39–0.80] |
| safe_oacbf | 59 | 27% [17–40] | 0% [0–6] | 15% [8–27] | 15% [8–27] | 37% [26–50] | 0.00 [0.00–0.00] |
| main:safe_oacbf | 59 | 29% [19–41] | 0% [0–6] | 22% [13–34] | 10% [5–20] | 37% [26–50] | 0.00 [0.00–0.00] |
| main:ant_nom | 59 | 81% [70–89] | 8% [4–18] | 20% [12–32] | 58% [45–69] | 0% [0–6] | 0.00 [0.00–0.00] |
| main:ant_cbf | 59 | 78% [66–87] | 7% [3–16] | 12% [6–23] | 61% [48–72] | 0% [0–6] | 0.00 [0.00–0.00] |

### Paired comparisons of ours (safe_oacbf@recover=reobserve)

Exact McNemar on the scene seeds both rows share. Holm over the four comparisons, per metric. `ours only` / `other only` = scenes where the event happened in that row only. `same target` = scenes where both rows grasped the same object.

| metric | vs | batch | n | ours | other | ours only | other only | same target | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| success | safe_oacbf | same | 224 | 40% | 29% | 33 | 8 | 224 | 0.00011 | 0.00022 |
| success | main:safe_oacbf | different | 224 | 40% | 32% | 45 | 27 | 177 | 0.044 | 0.044 |
| success | main:ant_nom | different | 224 | 40% | 60% | 21 | 66 | 177 | 1.4e-06 | 5.6e-06 |
| success | main:ant_cbf | different | 224 | 40% | 56% | 22 | 59 | 177 | 4.8e-05 | 0.00014 |
| collision | safe_oacbf | same | 224 | 1% | 1% | 1 | 0 | 224 | 1 | 1 |
| collision | main:safe_oacbf | different | 224 | 1% | 1% | 3 | 2 | 177 | 1 | 1 |
| collision | main:ant_nom | different | 224 | 1% | 8% | 2 | 16 | 177 | 0.0013 | 0.0052 |
| collision | main:ant_cbf | different | 224 | 1% | 5% | 2 | 11 | 177 | 0.022 | 0.067 |
| any_disturbed | safe_oacbf | same | 224 | 16% | 13% | 7 | 2 | 224 | 0.18 | 0.54 |
| any_disturbed | main:safe_oacbf | different | 224 | 16% | 17% | 22 | 26 | 177 | 0.67 | 0.7 |
| any_disturbed | main:ant_nom | different | 224 | 16% | 22% | 18 | 33 | 177 | 0.049 | 0.2 |
| any_disturbed | main:ant_cbf | different | 224 | 16% | 19% | 24 | 32 | 177 | 0.35 | 0.7 |
| clean_success | safe_oacbf | same | 224 | 34% | 23% | 32 | 7 | 224 | 7e-05 | 0.00028 |
| clean_success | main:safe_oacbf | different | 224 | 34% | 22% | 47 | 19 | 177 | 0.00076 | 0.0023 |
| clean_success | main:ant_nom | different | 224 | 34% | 46% | 28 | 53 | 177 | 0.0073 | 0.015 |
| clean_success | main:ant_cbf | different | 224 | 34% | 43% | 28 | 48 | 177 | 0.029 | 0.029 |
| deadlock | safe_oacbf | same | 224 | 4% | 26% | 0 | 51 | 224 | 8.9e-16 | 3.6e-15 |
| deadlock | main:safe_oacbf | different | 224 | 4% | 26% | 5 | 56 | 177 | 5.6e-12 | 1.7e-11 |
| deadlock | main:ant_nom | different | 224 | 4% | 1% | 8 | 3 | 177 | 0.23 | 0.45 |
| deadlock | main:ant_cbf | different | 224 | 4% | 1% | 8 | 3 | 177 | 0.23 | 0.45 |

### Per clutter level (unadjusted p)

| kind | metric | vs | n | ours | other | ours only | other only | p |
|---|---|---|---|---|---|---|---|---|
| sparse | success | safe_oacbf | 60 | 38% | 33% | 6 | 3 | 0.51 |
| sparse | success | main:ant_nom | 60 | 38% | 62% | 3 | 17 | 0.0026 |
| sparse | collision | safe_oacbf | 60 | 2% | 2% | 0 | 0 | 1 |
| sparse | collision | main:ant_nom | 60 | 2% | 5% | 0 | 2 | 0.5 |
| sparse | deadlock | safe_oacbf | 60 | 2% | 20% | 0 | 11 | 0.00098 |
| sparse | deadlock | main:ant_nom | 60 | 2% | 2% | 1 | 1 | 1 |
| medium | success | safe_oacbf | 57 | 35% | 30% | 5 | 2 | 0.45 |
| medium | success | main:ant_nom | 57 | 35% | 51% | 7 | 16 | 0.093 |
| medium | collision | safe_oacbf | 57 | 0% | 0% | 0 | 0 | 1 |
| medium | collision | main:ant_nom | 57 | 0% | 4% | 0 | 2 | 0.5 |
| medium | deadlock | safe_oacbf | 57 | 2% | 21% | 0 | 11 | 0.00098 |
| medium | deadlock | main:ant_nom | 57 | 2% | 2% | 1 | 1 | 1 |
| dense | success | safe_oacbf | 48 | 31% | 23% | 5 | 1 | 0.22 |
| dense | success | main:ant_nom | 48 | 31% | 42% | 6 | 11 | 0.33 |
| dense | collision | safe_oacbf | 48 | 2% | 2% | 0 | 0 | 1 |
| dense | collision | main:ant_nom | 48 | 2% | 15% | 1 | 7 | 0.07 |
| dense | deadlock | safe_oacbf | 48 | 10% | 27% | 0 | 8 | 0.0078 |
| dense | deadlock | main:ant_nom | 48 | 10% | 2% | 5 | 1 | 0.22 |
| hidden | success | safe_oacbf | 59 | 53% | 27% | 17 | 2 | 0.00073 |
| hidden | success | main:ant_nom | 59 | 53% | 81% | 5 | 22 | 0.0015 |
| hidden | collision | safe_oacbf | 59 | 2% | 0% | 1 | 0 | 1 |
| hidden | collision | main:ant_nom | 59 | 2% | 8% | 1 | 5 | 0.22 |
| hidden | deadlock | safe_oacbf | 59 | 2% | 37% | 0 | 21 | 9.5e-07 |
| hidden | deadlock | main:ant_nom | 59 | 2% | 0% | 1 | 0 | 1 |

### Run-to-run drift

The anchor `safe_oacbf` re-run here and the main batch's `safe_oacbf` have the same outcome in 142 of 224 scenes and the same target object in 177 of 224. Totals: success 64 vs 71, deadlock 59 vs 59, collision 2 vs 2.

### Noise inside this batch

In the 156 scenes where ours fired no recovery, both arms ran identical code. They still differ on success in 6 scenes (anchor only 6, ours only 0), on collision in 0, on deadlock in 0 and on disturbed ≥1 in 4.

### What became of the anchor's deadlocks in ours

deadlock: 8, slip: 19, success: 32 (of 59).

### Rows in the report's format (rates in %, Wilson 95% interval)

```latex
\texttt{safe\_oacbf} + re-observe & 40\,[34--46] & 1\,[0--4] & 16\,[11--21] & 34\,[28--41] & 4\,[2--7] \\
\texttt{safe\_oacbf} (same batch) & 29\,[23--35] & 1\,[0--3] & 13\,[10--18] & 23\,[18--29] & 26\,[21--32] \\
\texttt{safe\_oacbf} + re-observe & 38 & 35 & 31 & 53 \\
\texttt{safe\_oacbf} (same batch) & 33 & 30 & 23 & 27 \\
```
