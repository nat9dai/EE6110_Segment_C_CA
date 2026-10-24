"""Turn benchmark JSONL rows into summary tables (JSON, Markdown) and figures.

    python3 scripts/analyze.py --exp main --exp abl
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# colour follows the method, never its rank
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METHOD_ORDER = ["safe_oacbf", "td_nom", "ant_nom", "ant_apf", "ant_cbf", "ant_oacbf", "safe_nom", "safe_cbf"]
LABELS = {
    "td_nom": "Top-down + nominal", "ant_nom": "Antipodal + nominal", "ant_apf": "Antipodal + APF",
    "ant_cbf": "Antipodal + CBF", "ant_oacbf": "Antipodal + OA-CBF", "safe_nom": "Safety-ranked + nominal",
    "safe_cbf": "Safety-ranked + CBF", "safe_oacbf": "Safety-ranked + OA-CBF (ours)",
}
KIND_ORDER = ["sparse", "medium", "dense", "hidden"]


def contaminated_scenes(exp="main"):
    """Return seeds of scenes where the robot already touches another object at the start."""
    flagged = set()
    tdir = os.path.join(ROOT, "results", "raw", exp, "traces")
    for p in glob.glob(os.path.join(tdir, "*.npy")):
        T = np.load(p)
        if T.ndim == 2 and T.shape[1] >= 17 and len(T) and (T[0, 16] > 1.0 or T[0, 15] < 0.0):
            flagged.add(int(os.path.basename(p).split("_", 1)[0]))
    by = defaultdict(list)
    for p in glob.glob(os.path.join(ROOT, "results", "raw", exp, "w*.jsonl")):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get("skipped"):
                by[r["scene_seed"]].append(r)
    for s, rs in by.items():
        ex = [r for r in rs if r.get("phase_reached") not in ("plan", "error")]
        if ex and all(r.get("collision") and r.get("first_contact_phase") == "transit"
                      and r.get("min_gt_clearance_phase") == "transit" and (r.get("min_gt_clearance") or 0) < -0.005 for r in ex):
            flagged.add(s)
    # traces are not shipped, so save the list for a fresh copy to reuse
    store = os.path.join(ROOT, "results", "excluded_scenes.json")
    saved = json.load(open(store)) if os.path.exists(store) else {}
    flagged |= set(saved.get(exp, []))
    if flagged:
        saved[exp] = sorted(flagged)
        json.dump(saved, open(store, "w"), indent=1)
    return flagged


EXCLUDED = {}


def load(exp):
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "raw", exp, "w*.jsonl"))):
        with open(p) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("skipped"):
                    continue
                rows.append(r)
    # restarts can duplicate rows
    d = {(r["scene_seed"], r["method"]): r for r in rows}
    rows = list(d.values())
    # ablation scenes are a subset of main, so the main traces flag them too
    bad = contaminated_scenes("main") | contaminated_scenes(exp)
    kinds = {r["scene_seed"]: r.get("kind") for r in rows}
    errs = [r for r in rows if r.get("phase_reached") == "error"]
    EXCLUDED[exp] = dict(n_scenes=len({s for s in kinds if s in bad}),
                         by_kind={k: sum(1 for s, kk in kinds.items() if s in bad and kk == k) for k in KIND_ORDER},
                         n_error_rows=len(errs))
    rows = [r for r in rows if r["scene_seed"] not in bad and r.get("phase_reached") != "error"]
    for r in rows:  # held >= 8 cm up counts as success; the width check fails on thin parts
        if not r.get("success") and r.get("fail") == "closed_empty" and (r.get("target_lift") or 0) > 0.08:
            r["success"], r["fail"] = True, None
    for r in rows:
        r["clean_success"] = bool(r.get("success")) and not r.get("collision") and int(r.get("n_disturbed") or 0) == 0
    return rows


def holm(pvals):
    """Holm-Bonferroni adjusted p-values for a dict name -> p."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def boot_mean(x, n=2000, seed=0):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return (float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def mcnemar(a, b):
    """Exact two-sided McNemar p-value for paired booleans a, b."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    n = n01 + n10
    if n == 0:
        return 1.0, n01, n10
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p), n01, n10


def is_deadlock(r):
    f = r.get("fail") or ""
    return f.endswith("stall") or f.endswith("timeout")


def summarize(rows, key_fn):
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    out = {}
    for k, rs in groups.items():
        n = len(rs)
        succ = [bool(r.get("success")) for r in rs]
        coll = [bool(r.get("collision")) for r in rs]
        dist = [int(r.get("n_disturbed") or 0) for r in rs]
        safe_succ = [s and not c and d == 0 for s, c, d in zip(succ, coll, dist)]
        dead = [is_deadlock(r) for r in rs]
        fails = defaultdict(int)
        for r in rs:
            fails[r.get("fail") or "success"] += 1
        clear = [r.get("min_gt_clearance") for r in rs if r.get("phase_reached") not in ("plan", "error")]
        out[k] = dict(
            n=n,
            success=wilson(sum(succ), n), collision=wilson(sum(coll), n),
            any_disturbed=wilson(sum(d > 0 for d in dist), n), clean_success=wilson(sum(safe_succ), n),
            deadlock=wilson(sum(dead), n),
            n_disturbed=boot_mean(dist), max_force=boot_mean([r.get("max_contact_force") for r in rs]),
            min_clearance=boot_mean(clear),
            target_push=boot_mean([r.get("target_push") for r in rs]),
            qp_ms=boot_mean([r.get("qp_ms_mean") for r in rs if r.get("qp_ms_mean")]),
            qp_ms_max=float(np.max([r.get("qp_ms_max") or 0 for r in rs])),
            plan_s=boot_mean([(r.get("perc_perceive_s") or 0) + (r.get("propose_s") or 0) for r in rs]),
            exec_s=boot_mean([r.get("exec_sim_s") for r in rs if r.get("exec_sim_s")]),
            fails=dict(fails),
            collision_exec=wilson(sum(1 for r in rs if r.get("collision") and r.get("phase_reached") not in ("plan", "error")),
                                  max(1, sum(1 for r in rs if r.get("phase_reached") not in ("plan", "error")))),
            indirect_disturbance=wilson(sum(1 for c, d in zip(coll, dist) if d > 0 and not c), n),
            collision_phase={ph: sum(1 for r in rs if r.get("collision") and r.get("first_contact_phase") == ph) / n
                             for ph in ("transit", "approach", "close", "lift", "hold", "retreat")},
            attempts_mean=float(np.mean([r.get("n_attempts") or 0 for r in rs])),
        )
    return out


def paired(rows, ref, metric):
    by = defaultdict(dict)
    for r in rows:
        by[r["scene_seed"]][r["method"]] = r
    res = {}
    methods = sorted({r["method"] for r in rows})
    for m in methods:
        if m == ref:
            continue
        a, b = [], []
        for s, d in by.items():
            if ref in d and m in d:
                a.append(bool(d[ref].get(metric)))
                b.append(bool(d[m].get(metric)))
        if a:
            p, n01, n10 = mcnemar(a, b)
            res[m] = dict(n=len(a), ref_rate=float(np.mean(a)), other_rate=float(np.mean(b)), p=p,
                          ref_only=n10, other_only=n01)
    return res


def fmt_rate(t):
    p, lo, hi = t
    return "–" if math.isnan(p) else f"{100 * p:.0f}% [{100 * lo:.0f}–{100 * hi:.0f}]"


def fmt_mean(t, scale=1.0, dec=2):
    m, lo, hi = t
    return "–" if math.isnan(m) else f"{m * scale:.{dec}f} [{lo * scale:.{dec}f}–{hi * scale:.{dec}f}]"


def method_sort(m):
    base = m.split("@")[0]
    return (METHOD_ORDER.index(base) if base in METHOD_ORDER else 99, m)


def make_figures(exp, rows, summ_mk, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                         "axes.axisbelow": True})
    import plot_style
    plot_style.apply()
    methods = sorted({r["method"] for r in rows}, key=method_sort)
    kinds = [k for k in KIND_ORDER if any(r.get("kind") == k for r in rows)]
    color = {m: PALETTE[METHOD_ORDER.index(m.split("@")[0]) % 8] if m.split("@")[0] in METHOD_ORDER else "#8a8984" for m in methods}
    for metric, title in [("success", "Grasp success rate"), ("collision", "Collateral collision rate"),
                          ("clean_success", "Clean success (success, no collision, nothing disturbed)"),
                          ("deadlock", "Deadlock / timeout rate")]:
        fig, ax = plt.subplots(figsize=(7.2, 3.0))
        w = 0.8 / max(1, len(methods))
        for j, m in enumerate(methods):
            xs, ys, lo, hi = [], [], [], []
            for i, k in enumerate(kinds):
                s = summ_mk.get((m, k))
                if not s:
                    continue
                p, l, h = s[metric]
                xs.append(i - 0.4 + w * (j + 0.5))
                ys.append(100 * p)
                lo.append(max(0.0, 100 * (p - l)))
                hi.append(max(0.0, 100 * (h - p)))
            ax.bar(xs, ys, width=w * 0.9, color=color[m], label=LABELS.get(m, m), edgecolor="white", linewidth=0.8)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor="#52514e", elinewidth=0.8, capsize=1.5)
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels(kinds)
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
        ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b")
        ax.legend(fontsize=7, ncol=2, frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{exp}_{metric}.png"), dpi=160)
        plt.close(fig)
    summ_m = summarize(rows, lambda r: r["method"])
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, ym, yl in [(axs[0], "success", "success rate [%]"), (axs[1], "clean_success", "clean success [%]")]:
        for m in methods:
            sm = summ_m[m]
            x, xl, xh = [100 * v for v in sm["collision"]]
            y, ylo, yhi = [100 * v for v in sm[ym]]
            ax.errorbar(x, y, xerr=[[x - xl], [xh - x]], yerr=[[y - ylo], [yhi - y]], fmt="o", color=color[m], ms=7,
                        elinewidth=0.8, capsize=2, mec="white", mew=1.2, label=LABELS.get(m, m))
            ax.annotate(m, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=7, color="#52514e")
        ax.set_xlabel("collateral collision rate [%]  (lower is safer)")
        ax.set_ylabel(yl)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 100)
    axs[1].legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{exp}_pareto.png"), dpi=160)
    plt.close(fig)
    cats = ["success", "no grasp / IK", "transit fail", "approach deadlock", "slip / empty", "error"]

    def cat(r):
        f = r.get("fail")
        if r.get("success"):
            return "success"
        if f is None and r.get("phase_reached") == "error":
            return "error"
        f = f or ""
        if f in ("no_grasp", "ik_fail", "target_not_seen"):
            return "no grasp / IK"
        if f.startswith("transit"):
            return "transit fail"
        if f.startswith("approach"):
            return "approach deadlock"
        if f in ("slip", "closed_empty"):
            return "slip / empty"
        return "error"
    ccol = ["#2a78d6", "#8a8984", "#eda100", "#eb6834", "#e87ba4", "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(methods) + 1.0))
    for j, m in enumerate(methods):
        rs = [r for r in rows if r["method"] == m]
        left = 0.0
        for c, col in zip(cats, ccol):
            v = 100 * sum(cat(r) == c for r in rs) / max(1, len(rs))
            ax.barh(j, v, left=left, color=col, edgecolor="white", linewidth=1.0, label=c if j == 0 else None)
            left += v
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([LABELS.get(m, m) for m in methods], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of trials")
    ax.set_title("Outcome breakdown (all scene types)", loc="left", fontsize=10)
    ax.legend(fontsize=7, ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.12), loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{exp}_outcomes.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", action="append", default=None)
    ap.add_argument("--ref", default="safe_oacbf")
    a = ap.parse_args()
    exps = a.exp or ["main"]
    fig_dir = os.path.join(ROOT, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for exp in exps:
        rows = load(exp)
        if not rows:
            print(f"[{exp}] no rows")
            continue
        s_mk = summarize(rows, lambda r: (r["method"], r.get("kind")))
        s_m = summarize(rows, lambda r: r["method"])
        pairs = {k: {met: paired([r for r in rows if r.get("kind") == k], a.ref, met) for met in ("success", "collision")}
                 for k in KIND_ORDER}
        pairs["all"] = {met: paired(rows, a.ref, met) for met in ("success", "collision", "clean_success")}
        for met, dd in pairs["all"].items():
            adj = holm({m: v["p"] for m, v in dd.items()})
            for m in dd:
                dd[m]["p_holm"] = adj[m]
        out = dict(exp=exp, excluded=EXCLUDED.get(exp), n_rows=len(rows), n_scenes=len({r["scene_seed"] for r in rows}),
                   n_objects_used=len({o for r in rows for o in r.get("objects", [])}),
                   by_method={m: v for m, v in s_m.items()},
                   by_method_kind={f"{m}|{k}": v for (m, k), v in s_mk.items()}, paired_vs_ref=pairs, ref=a.ref)
        with open(os.path.join(ROOT, "results", f"summary_{exp}.json"), "w") as f:
            json.dump(out, f, indent=1)
        methods = sorted(s_m, key=method_sort)
        lines = [f"## {exp}: {len(rows)} trials, {out['n_scenes']} scenes, {out['n_objects_used']} distinct GSO objects", "",
                 "| method | n | success | collision | disturbed≥1 | clean success | deadlock | #disturbed | min GT clearance [mm] | QP [ms] |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for m in methods:
            s = s_m[m]
            lines.append(f"| {m} | {s['n']} | {fmt_rate(s['success'])} | {fmt_rate(s['collision'])} | {fmt_rate(s['any_disturbed'])} | "
                         f"{fmt_rate(s['clean_success'])} | {fmt_rate(s['deadlock'])} | {fmt_mean(s['n_disturbed'])} | "
                         f"{fmt_mean(s['min_clearance'], 1000, 1)} | {fmt_mean(s['qp_ms'], 1, 2)} |")
        for k in KIND_ORDER:
            ms = [m for m in methods if (m, k) in s_mk]
            if not ms:
                continue
            lines += ["", f"### {k}", "", "| method | n | success | collision | disturbed≥1 | clean success | deadlock |", "|---|---|---|---|---|---|---|"]
            for m in ms:
                s = s_mk[(m, k)]
                lines.append(f"| {m} | {s['n']} | {fmt_rate(s['success'])} | {fmt_rate(s['collision'])} | {fmt_rate(s['any_disturbed'])} | "
                             f"{fmt_rate(s['clean_success'])} | {fmt_rate(s['deadlock'])} |")
        lines += ["", f"### paired McNemar vs {a.ref} (all kinds)", "", "| other | n | metric | ref | other | p |", "|---|---|---|---|---|---|"]
        for met in ("success", "collision"):
            for m, v in sorted(pairs["all"][met].items(), key=lambda kv: method_sort(kv[0])):
                lines.append(f"| {m} | {v['n']} | {met} | {100 * v['ref_rate']:.0f}% | {100 * v['other_rate']:.0f}% | {v['p']:.2g} |")
        md = "\n".join(lines)
        with open(os.path.join(ROOT, "results", f"summary_{exp}.md"), "w") as f:
            f.write(md + "\n")
        print(md)
        try:
            make_figures(exp, rows, s_mk, fig_dir)
        except Exception as e:  # noqa: BLE001
            print("figure error", repr(e))


if __name__ == "__main__":
    main()
