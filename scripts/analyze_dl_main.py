"""Summarise deadlock recovery on the main scenes (results/raw/dl_main) into results/summary_dl_main.{md,json}.

    python3 scripts/analyze_dl_main.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import analyze as an
import analyze_dl as dl

EXP = "dl_main"
PLATFORM = "Isaac Sim 5.1"
OURS = "safe_oacbf@recover=reobserve"
ANCHOR = "safe_oacbf"
# 'main:' rows come from the main benchmark batch, a different run
COMPARE = [ANCHOR, "main:safe_oacbf", "main:ant_nom", "main:ant_cbf"]
METRICS = ("success", "collision", "any_disturbed", "clean_success", "deadlock")
KINDS = an.KIND_ORDER


def prep(r):
    r["deadlock"] = an.is_deadlock(r)
    r["any_disturbed"] = int(r.get("n_disturbed") or 0) > 0
    r["recoveries"] = int(r.get("recoveries") or 0)
    r["recoveries_effective"] = sum(dl.changed(e) for e in r.get("recovery_events") or [])
    r["outcome"] = dl.outcome(r)
    return r


def load():
    rows = [prep(r) for r in an.load(EXP) if r["method"] in (OURS, ANCHOR)]
    by = defaultdict(dict)
    for r in rows:
        by[r["scene_seed"]][r["method"]] = r
    main = [prep(dict(r, method="main:" + r["method"])) for r in an.load("main")]
    mby = defaultdict(dict)
    for r in main:
        mby[r["scene_seed"]][r["method"]] = r
    both = {s for s, d in by.items() if OURS in d and ANCHOR in d}
    scenes = sorted(s for s in both if s in mby)
    store = json.load(open(os.path.join(an.ROOT, "results", "excluded_scenes.json")))
    info = dict(excluded=sorted(store.get("main", [])), excluded_batch=sorted(store.get(EXP, [])),
                incomplete_in_batch=sorted(set(by) - both), not_in_main=sorted(both - set(mby)),
                main_not_in_batch=sorted(set(mby) - both))
    keep = set(scenes)
    rows = [r for r in rows if r["scene_seed"] in keep]
    main = [r for r in main if r["scene_seed"] in keep]
    return rows, main, info


def paired(rs, metric, other):
    """Compare ours with ``other`` on the scenes both ran (rates, discordant counts, McNemar p)."""
    by = defaultdict(dict)
    for r in rs:
        by[r["scene_seed"]][r["method"]] = r
    a, b, same_target = [], [], 0
    for d in by.values():
        if OURS in d and other in d:
            a.append(bool(d[OURS][metric]))
            b.append(bool(d[other][metric]))
            same_target += d[OURS].get("target_name") == d[other].get("target_name")
    p, n01, n10 = an.mcnemar(b, a)  # n10 = other only, n01 = ours only
    k = max(1, len(a))  # partial runs can leave a clutter level empty
    return dict(n=len(a), ours=sum(a) / k, other=sum(b) / k, ours_only=n01, other_only=n10, p=p,
                same_target=same_target)


def main():
    rows, main_rows, info = load()
    allr = rows + main_rows
    methods = [OURS, ANCHOR] + ["main:" + m for m in an.METHOD_ORDER]
    summ = {m: dl.arm_summary([r for r in allr if r["method"] == m]) for m in methods}
    summ_k = {k: {m: dl.arm_summary([r for r in allr if r["method"] == m and r.get("kind") == k]) for m in methods}
              for k in KINDS}
    pairs = {met: {o: paired(allr, met, o) for o in COMPARE} for met in METRICS}
    for met in METRICS:
        adj = an.holm({o: v["p"] for o, v in pairs[met].items()})
        for o in pairs[met]:
            pairs[met][o]["p_holm"] = adj[o]
    pairs_k = {k: {met: {o: paired([r for r in allr if r.get("kind") == k], met, o) for o in COMPARE}
                   for met in ("success", "collision", "deadlock")} for k in KINDS}
    by = defaultdict(dict)
    for r in allr:
        by[r["scene_seed"]][r["method"]] = r
    drift = dict(n=len(by), same_outcome=sum(d[ANCHOR]["outcome"] == d["main:safe_oacbf"]["outcome"] for d in by.values()),
                 same_target=sum(d[ANCHOR].get("target_name") == d["main:safe_oacbf"].get("target_name") for d in by.values()))
    n = len(by)
    # no recovery fired, so both arms ran the same code: any difference is physics noise
    quiet = [d for d in by.values() if d[OURS]["recoveries"] == 0]
    noise = dict(n=len(quiet), **{f"diff_{k}": sum(bool(d[OURS][k]) != bool(d[ANCHOR][k]) for d in quiet)
                                  for k in ("success", "collision", "deadlock", "any_disturbed")})
    noise["anchor_only_success"] = sum(bool(d[ANCHOR]["success"]) and not d[OURS]["success"] for d in quiet)
    noise["ours_only_success"] = sum(bool(d[OURS]["success"]) and not d[ANCHOR]["success"] for d in quiet)
    rescued = dict(defaultdict(int))
    for d in by.values():
        if d[ANCHOR]["deadlock"]:
            rescued[d[OURS]["outcome"]] = rescued.get(d[OURS]["outcome"], 0) + 1
    out = dict(exp=EXP, platform=PLATFORM, ours=OURS, anchor=ANCHOR, n_scenes=n, n_rows_batch=len(rows),
               scenes=info, summary=summ, summary_by_kind=summ_k, paired=pairs, paired_by_kind=pairs_k, drift=drift,
               noise_same_batch=noise, anchor_deadlocks_in_ours=rescued,
               note="'main:' rows come from the main benchmark batch (results/raw/main), a different run. Holm is over "
                    "the four comparisons of ours, per metric. Per-kind p-values are unadjusted.")
    with open(os.path.join(an.ROOT, "results", f"summary_{EXP}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)

    L = [f"## Deadlock recovery on the main benchmark scenes ({PLATFORM}): {len(rows)} trials, {n} scenes x 2 arms", "",
         f"Same launcher and flags as scripts/bench_main.sh (pool seeds 7, 8, 9; 5 workers; 20 scenes per kind; main "
         f"camera only). Excluded: the {len(info['excluded'])} 'main' scenes of results/excluded_scenes.json, plus this "
         f"batch's own audit flags {info['excluded_batch']}. "
         f"Dropped: incomplete in this batch {info['incomplete_in_batch']}, not in the main rows {info['not_in_main']}, "
         f"main scenes missing here {info['main_not_in_batch']}.",
         "", "Rows marked `main:` are the main benchmark's own trials on the same scene seeds, from a different batch.", "",
         "### All scenes", "",
         "| method | n | success | collision | disturbed ≥1 | clean success | deadlock | recoveries / trial | trials with a recovery |",
         "|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        s = summ[m]
        L.append(f"| {m} | {s['n']} | {an.fmt_rate(s['success'])} | {an.fmt_rate(s['collision'])} | {an.fmt_rate(s['any_disturbed'])} | "
                 f"{an.fmt_rate(s['clean_success'])} | {an.fmt_rate(s['deadlock'])} | {an.fmt_mean(s['recoveries_mean'])} | "
                 f"{an.fmt_rate(s['recovered_trials'])} |")
    L += ["", "### Success by clutter level (%)", "",
          "| method | " + " | ".join(f"{k} (n = {summ_k[k][OURS]['n']})" for k in KINDS) + " |", "|---|" + "---|" * len(KINDS)]
    for m in methods:
        L.append(f"| {m} | " + " | ".join(f"{100 * summ_k[k][m]['success'][0]:.0f}" for k in KINDS) + " |")
    for k in KINDS:
        L += ["", f"### {k}", "", "| method | n | success | collision | disturbed ≥1 | clean success | deadlock | recoveries / trial |",
              "|---|---|---|---|---|---|---|---|"]
        for m in (OURS, ANCHOR, "main:safe_oacbf", "main:ant_nom", "main:ant_cbf"):
            s = summ_k[k][m]
            L.append(f"| {m} | {s['n']} | {an.fmt_rate(s['success'])} | {an.fmt_rate(s['collision'])} | {an.fmt_rate(s['any_disturbed'])} | "
                     f"{an.fmt_rate(s['clean_success'])} | {an.fmt_rate(s['deadlock'])} | {an.fmt_mean(s['recoveries_mean'])} |")
    L += ["", f"### Paired comparisons of ours ({OURS})", "",
          "Exact McNemar on the scene seeds both rows share. Holm over the four comparisons, per metric. "
          "`ours only` / `other only` = scenes where the event happened in that row only. "
          "`same target` = scenes where both rows grasped the same object.", "",
          "| metric | vs | batch | n | ours | other | ours only | other only | same target | p | p (Holm) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for met in METRICS:
        for o in COMPARE:
            v = pairs[met][o]
            batch = "same" if o == ANCHOR else "different"
            L.append(f"| {met} | {o} | {batch} | {v['n']} | {100 * v['ours']:.0f}% | {100 * v['other']:.0f}% | {v['ours_only']} | "
                     f"{v['other_only']} | {v['same_target']} | {v['p']:.2g} | {v['p_holm']:.2g} |")
    L += ["", "### Per clutter level (unadjusted p)", "", "| kind | metric | vs | n | ours | other | ours only | other only | p |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in KINDS:
        for met in ("success", "collision", "deadlock"):
            for o in (ANCHOR, "main:ant_nom"):
                v = pairs_k[k][met][o]
                L.append(f"| {k} | {met} | {o} | {v['n']} | {100 * v['ours']:.0f}% | {100 * v['other']:.0f}% | "
                         f"{v['ours_only']} | {v['other_only']} | {v['p']:.2g} |")
    L += ["", "### Run-to-run drift", "",
          f"The anchor `{ANCHOR}` re-run here and the main batch's `safe_oacbf` have the same outcome in "
          f"{drift['same_outcome']} of {n} scenes and the same target object in {drift['same_target']} of {n}. "
          f"Totals: success {round(summ[ANCHOR]['success'][0] * n)} vs {round(summ['main:safe_oacbf']['success'][0] * n)}, "
          f"deadlock {round(summ[ANCHOR]['deadlock'][0] * n)} vs {round(summ['main:safe_oacbf']['deadlock'][0] * n)}, "
          f"collision {round(summ[ANCHOR]['collision'][0] * n)} vs {round(summ['main:safe_oacbf']['collision'][0] * n)}.",
          "", "### Noise inside this batch", "",
          f"In the {noise['n']} scenes where ours fired no recovery, both arms ran identical code. They still differ on "
          f"success in {noise['diff_success']} scenes (anchor only {noise['anchor_only_success']}, ours only "
          f"{noise['ours_only_success']}), on collision in {noise['diff_collision']}, on deadlock in {noise['diff_deadlock']} "
          f"and on disturbed ≥1 in {noise['diff_any_disturbed']}.",
          "", "### What became of the anchor's deadlocks in ours", "",
          ", ".join(f"{k}: {v}" for k, v in sorted(rescued.items())) + f" (of {sum(rescued.values())}).",
          "", "### Rows in the report's format (rates in %, Wilson 95% interval)", "", "```latex"]
    def cell(t):
        return f"{100 * t[0]:.0f}\\,[{100 * t[1]:.0f}--{100 * t[2]:.0f}]"
    for m, lab in ((OURS, "\\texttt{safe\\_oacbf} + re-observe"), (ANCHOR, "\\texttt{safe\\_oacbf} (same batch)")):
        sm = summ[m]
        L.append(f"{lab} & " + " & ".join(cell(sm[k]) for k in METRICS) + " \\\\")
    for m, lab in ((OURS, "\\texttt{safe\\_oacbf} + re-observe"), (ANCHOR, "\\texttt{safe\\_oacbf} (same batch)")):
        L.append(f"{lab} & " + " & ".join(f"{100 * summ_k[k][m]['success'][0]:.0f}" for k in KINDS) + " \\\\")
    L += ["```"]
    md = "\n".join(L)
    with open(os.path.join(an.ROOT, "results", f"summary_{EXP}.md"), "w") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
