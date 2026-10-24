"""Summarise the deadlock-recovery A/B runs (results/raw/dl) into results/summary_dl.{md,json} and a figure.

    python3 scripts/analyze_dl.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import analyze as an

EXP = "dl"
PLATFORM = "Isaac Sim 5.1"
ARMS = {"A0": "safe_oacbf", "A1": "safe_oacbf@recover=reobserve", "A2": "safe_oacbf@recover=relax"}
ARM_LABEL = {"A0": "A0 reference", "A1": "A1 re-observe on stall", "A2": "A2 relax margin on stall"}
REF = ARMS["A0"]
METRICS = ("success", "deadlock", "collision", "any_disturbed", "clean_success")
KINDS = ("dense", "hidden")
# analyze.py palette, checked for colour-blind separation
CATS = [("success", "success", "#2a78d6"), ("slip", "slip / closed empty", "#4a3aa7"),
        ("deadlock", "deadlock (stall or timeout)", "#eb6834"), ("plan", "no grasp found", "#8a8984")]


def outcome(r):
    if r.get("success"):
        return "success"
    if an.is_deadlock(r):
        return "deadlock"
    if r.get("phase_reached") == "plan":
        return "plan"
    return "slip"


def changed(ev):
    """True if the recovery changed the barriers (a second event is usually a no-op)."""
    if ev["mode"] == "reobserve":
        return ev.get("n_unk_barrier_after") != ev.get("n_unk_barrier_before")
    return (ev.get("margin_unk_before") or 0.0) > 0.0


def load():
    rows = an.load(EXP)
    store = json.load(open(os.path.join(an.ROOT, "results", "excluded_scenes.json")))
    excl = set().union(*(set(v) for v in store.values()))
    raw_seeds = set()
    for p in os.listdir(os.path.join(an.ROOT, "results", "raw", EXP)):
        if p.startswith("w") and p.endswith(".jsonl"):
            for line in open(os.path.join(an.ROOT, "results", "raw", EXP, p)):
                try:
                    raw_seeds.add(json.loads(line)["scene_seed"])
                except (json.JSONDecodeError, KeyError):
                    pass
    excluded_here = sorted(raw_seeds & excl)
    rows = [r for r in rows if r["scene_seed"] not in excl and r["method"] in ARMS.values()]
    by = defaultdict(dict)
    for r in rows:
        by[r["scene_seed"]][r["method"]] = r
    complete = {s for s, d in by.items() if len(d) == len(ARMS)}
    incomplete = sorted(set(by) - complete)
    rows = [r for r in rows if r["scene_seed"] in complete]
    for r in rows:
        r["deadlock"] = an.is_deadlock(r)
        r["any_disturbed"] = int(r.get("n_disturbed") or 0) > 0
        r["recoveries"] = int(r.get("recoveries") or 0)
        r["recoveries_effective"] = sum(changed(e) for e in r.get("recovery_events") or [])
        r["outcome"] = outcome(r)
    skipped = sorted(raw_seeds - set(by) - excl)
    return rows, dict(excluded=excluded_here, incomplete=incomplete, no_target_or_error=skipped)


def arm_summary(rs):
    n = len(rs)
    s = {m: an.wilson(sum(bool(r[m]) for r in rs), n) for m in METRICS}
    s["n"] = n
    rec = [r["recoveries"] for r in rs]
    s["recoveries_mean"] = an.boot_mean(rec)
    s["recovered_trials"] = an.wilson(sum(x > 0 for x in rec), n)
    s["recoveries_effective_mean"] = an.boot_mean([r["recoveries_effective"] for r in rs])
    s["wall_s"] = an.boot_mean([r.get("wall_s") for r in rs])
    s["exec_sim_s"] = an.boot_mean([r.get("exec_sim_s") or 0.0 for r in rs])
    s["outcomes"] = {c: sum(r["outcome"] == c for r in rs) for c, _, _ in CATS}
    ev = [e for r in rs for e in r.get("recovery_events") or []]
    s["events"] = dict(n=len(ev), first_after_done=sum(e["k"] == 1 and e.get("after") == "done" for e in ev),
                       first=sum(e["k"] == 1 for e in ev), phase={p: sum(e["phase"] == p for e in ev) for p in ("transit", "approach")})
    return s


def paired(rows, metric):
    out = {}
    for a in ("A1", "A2"):
        res = an.paired([r for r in rows if r["method"] in (REF, ARMS[a])], REF, metric)[ARMS[a]]
        out[a] = res
    adj = an.holm({a: v["p"] for a, v in out.items()})
    for a in out:
        out[a]["p_holm"] = adj[a]
    return out


def pct(t):
    return an.fmt_rate(t)


def table(summ, title):
    L = [f"### {title}", "",
         "| arm | n | success | collision | disturbed ≥1 | clean success | deadlock | recoveries / trial | trials with a recovery | wall time / trial [s] |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        s = summ[a]
        L.append(f"| {ARM_LABEL[a]} | {s['n']} | {pct(s['success'])} | {pct(s['collision'])} | {pct(s['any_disturbed'])} | "
                 f"{pct(s['clean_success'])} | {pct(s['deadlock'])} | {an.fmt_mean(s['recoveries_mean'])} | "
                 f"{pct(s['recovered_trials'])} | {an.fmt_mean(s['wall_s'], 1, 1)} |")
    return L


def make_figure(summ_by, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                         "axes.axisbelow": True})
    import plot_style
    plot_style.apply()
    groups = [(g, gl) for g, gl in [("all", "all scenes"), ("dense", "dense"), ("hidden", "hidden")] if g in summ_by]
    fig, axs = plt.subplots(len(groups), 1, figsize=(7.2, 4.6), sharex=True)
    for ax, (g, gl) in zip(axs, groups):
        summ = summ_by[g]
        for j, a in enumerate(ARMS):
            s = summ[a]
            left = 0.0
            for c, cl, col in CATS:
                v = 100 * s["outcomes"][c] / s["n"]
                ax.barh(j, v, left=left, height=0.72, color=col, edgecolor="white", linewidth=1.5,
                        label=cl if (g == "all" and j == 0) else None)
                if v >= 7:
                    ax.text(left + v / 2, j, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="white")
                left += v
            ax.text(101, j, f"coll. {100 * s['collision'][0]:.0f}%", va="center", fontsize=7, color="#52514e")
        ax.set_yticks(range(len(ARMS)))
        ax.set_yticklabels([ARM_LABEL[a] for a in ARMS], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{gl} (n = {summ['A0']['n']} scenes per arm)", loc="left", fontsize=9, color="#0b0b0b")
    axs[-1].set_xlabel(f"% of trials  ({PLATFORM}, safe_oacbf, paired per scene)")
    axs[0].legend(fontsize=7.5, ncol=4, frameon=False, bbox_to_anchor=(0.0, 1.24), loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    rows, info = load()
    arm_of = {m: a for a, m in ARMS.items()}
    groups = [g for g in ("all",) + KINDS if any(g == "all" or r.get("kind") == g for r in rows)]
    summ_by = {}
    for g in groups:
        rs = [r for r in rows if g == "all" or r.get("kind") == g]
        summ_by[g] = {a: arm_summary([r for r in rs if r["method"] == m]) for a, m in ARMS.items()}
    pairs = {g: {met: paired([r for r in rows if g == "all" or r.get("kind") == g], met) for met in METRICS}
             for g in groups}
    n_scenes = len({r["scene_seed"] for r in rows})
    out = dict(exp=EXP, platform=PLATFORM, arms=ARMS, n_scenes=n_scenes, n_rows=len(rows),
               n_objects_used=len({o for r in rows for o in r.get("objects", [])}), scenes=info,
               summary=summ_by, paired_vs_A0=pairs,
               note="Holm is applied over the two comparisons (A1 vs A0, A2 vs A0) separately for each metric; "
                    "the per-kind p-values are unadjusted.")
    with open(os.path.join(an.ROOT, "results", f"summary_{EXP}.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)

    L = [f"## Deadlock recovery A/B ({PLATFORM}): {len(rows)} trials, {n_scenes} scenes x 3 arms, "
         f"{out['n_objects_used']} distinct GSO objects", "",
         f"Scenes: dense and hidden, pool seeds 7 and 8, 15 per kind each (the controller-ablation scenes). "
         f"Excluded by results/excluded_scenes.json: {len(info['excluded'])} {info['excluded']}. "
         f"No valid target or no row: {info['no_target_or_error']}. Incomplete (dropped): {info['incomplete']}.", ""]
    L += table(summ_by["all"], "All scenes")
    for k in groups[1:]:
        L += [""] + table(summ_by[k], k)
    L += ["", "### Paired McNemar vs A0 (exact, two-sided)", "",
          "Holm over the two comparisons, per metric (all scenes). `A0 only` / `arm only` = discordant scenes.", "",
          "| scenes | metric | arm | n | A0 | arm | A0 only | arm only | p | p (Holm) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g in groups:
        for met in ("success", "deadlock", "collision", "clean_success"):
            for a in ("A1", "A2"):
                v = pairs[g][met][a]
                ph = f"{v['p_holm']:.2g}" if g == "all" else "–"
                L.append(f"| {g} | {met} | {a} | {v['n']} | {100 * v['ref_rate']:.0f}% | {100 * v['other_rate']:.0f}% | "
                         f"{v['ref_only']} | {v['other_only']} | {v['p']:.2g} | {ph} |")
    L += ["", "### Recovery events (all scenes)", "", "| arm | events | in transit | in approach | effective (changed the barriers) "
          "| first event resumed to a full approach | outcomes (success / slip / deadlock / no grasp) |", "|---|---|---|---|---|---|---|"]
    for a in ARMS:
        s = summ_by["all"][a]
        e = s["events"]
        eff = sum(r["recoveries_effective"] for r in rows if arm_of[r["method"]] == a)
        o = s["outcomes"]
        L.append(f"| {ARM_LABEL[a]} | {e['n']} | {e['phase']['transit']} | {e['phase']['approach']} | {eff} | "
                 f"{e['first_after_done']} of {e['first']} | {o['success']} / {o['slip']} / {o['deadlock']} / {o['plan']} |")
    md = "\n".join(L)
    with open(os.path.join(an.ROOT, "results", f"summary_{EXP}.md"), "w") as f:
        f.write(md + "\n")
    print(md)
    fig_dir = os.path.join(an.ROOT, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    make_figure(summ_by, os.path.join(fig_dir, f"{EXP}_outcomes.png"))


if __name__ == "__main__":
    main()
