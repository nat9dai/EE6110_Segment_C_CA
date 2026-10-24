#!/usr/bin/env python
"""Convert Google Scanned Objects (data/gso_raw/models/*) into Isaac-ready USD files and a catalog.

    ~/isaacsim/python.sh scripts/convert_gso.py                    # all objects, 16 workers
    ~/isaacsim/python.sh scripts/convert_gso.py --names Foo Bar    # a subset, merged into the catalog
    ~/isaacsim/python.sh scripts/convert_gso.py --stats-only       # redo the stats from the catalog
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from safegrasp.assets import gso  # noqa: E402

RESULTS = REPO / "results"


def _parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=str(gso.RAW_DIR))
    ap.add_argument("--usd-dir", default=str(gso.USD_DIR))
    ap.add_argument("--points-dir", default=str(gso.POINTS_DIR))
    ap.add_argument("--catalog", default=str(gso.CATALOG_PATH))
    ap.add_argument("--validation", default=str(gso.VALIDATION_PATH), help="sim validation flags to merge")
    ap.add_argument("--names", nargs="*", default=None, help="convert only these objects")
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N objects (debug)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--format", choices=["usd", "usda"], default="usd")
    ap.add_argument("--flip-v", action="store_true", help="flip texture v (not needed for GSO)")
    ap.add_argument("--skip-existing", action="store_true", help="keep existing catalog rows whose files exist")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--no-stats", action="store_true")
    return ap.parse_args()


def _ensure_pxr(args):
    """Re-run this script with the omni.usd.libs environment so ``pxr`` imports without Kit."""
    if gso.pxr_available() or os.environ.get("_GSO_PXR_REEXEC"):
        return
    env = dict(os.environ)
    env.update(gso.pxr_env())
    env["_GSO_PXR_REEXEC"] = "1"
    print("[convert_gso] re-exec with omni.usd.libs pxr environment", flush=True)
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


_CFG: gso.ConvertCfg | None = None
_DIRS: tuple | None = None


def _init(cfg, dirs):
    global _CFG, _DIRS
    _CFG, _DIRS = cfg, dirs


def _work(name):
    t = time.time()
    try:
        e = gso.convert_object(name, *_DIRS, cfg=_CFG)
    except Exception as ex:  # noqa: BLE001
        e = dict(name=name, ok=False, notes=[f"exception: {ex!r}"])
    e["convert_s"] = round(time.time() - t, 3)
    return e


def make_stats(rows: list[dict], fig_path: Path, json_path: Path) -> dict:
    """Write the dataset statistics (JSON) and their figure; return the stats."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    conv = [r for r in rows if r.get("extents_sorted") is not None and r.get("mass") is not None]
    ok = [r for r in conv if r.get("ok")]
    E = np.array([r["extents_sorted"] for r in ok])
    Eraw = np.array([sorted(r["extents_raw"]) for r in ok])
    M = np.array([r["mass"] for r in ok])
    Vol = np.array([r["volume"] for r in ok])
    H = np.array([r["n_hulls"] for r in ok])
    S = np.array([r["scale"] for r in ok])
    pct = [0, 5, 25, 50, 75, 95, 100]

    def P(a):
        return {f"p{p}": float(v) for p, v in zip(pct, np.percentile(a, pct))}

    fail_reasons: dict[str, int] = {}
    for r in rows:
        if not r.get("ok"):
            note = r["notes"][-1] if r.get("notes") else "unknown"
            if note.startswith("sim: "):  # "sim: sinks: bottom z = ..." -> "sim: sinks"
                key = "sim: " + note[5:].split(":")[0]
            else:  # strip numbers so equal reasons group
                key = re.sub(r"[-+]?\d+(\.\d+)?", "#", note.split(":")[0] if "extent" not in note else note)
            fail_reasons[key] = fail_reasons.get(key, 0) + 1
    stats = dict(
        n_total=len(rows),
        n_converted=len(conv),
        n_ok=len(ok),
        n_not_ok=len(rows) - len(ok),
        not_ok_reasons=fail_reasons,
        n_sim_validated=int(sum(bool(r.get("sim_validated")) for r in rows)),
        n_sim_failed=int(sum(r.get("sim_validated", False) and not r.get("sim_ok", True) for r in rows)),
        frac_scaled=float(np.mean(S < 1.0 - 1e-9)),
        frac_scaled_by_min_extent=float(np.mean(np.isclose(E[:, 0], 0.065, atol=1e-6))),
        frac_scaled_by_max_extent=float(np.mean(np.isclose(E[:, 2], 0.22, atol=1e-6))),
        scale=P(S),
        extent_min=P(E[:, 0]),
        extent_mid=P(E[:, 1]),
        extent_max=P(E[:, 2]),
        extent_raw_min=P(Eraw[:, 0]),
        extent_raw_max=P(Eraw[:, 2]),
        mass=P(M),
        frac_mass_clipped_low=float(np.mean(M <= 0.03 + 1e-9)),
        frac_mass_clipped_high=float(np.mean(M >= 0.5 - 1e-9)),
        volume=P(Vol),
        n_hulls_hist={int(k): int(v) for k, v in zip(*np.unique(H, return_counts=True))},
        mass_model="mass = clip(350 kg/m^3 * V_union_hulls, 0.03, 0.5); V on 64^3 grid",
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(stats, indent=1))

    c1, c2, c3 = "#2a78d6", "#eb6834", "#1baf7a"
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.titlesize": 10, "axes.titleweight": "bold"})  # fmt: skip
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import plot_style
    plot_style.apply()
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.2), facecolor="#fcfcfb")
    ax = ax.ravel()
    for a in ax:
        a.set_facecolor("#fcfcfb")
        a.grid(True, color=grid, lw=0.6)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    bins = np.linspace(0, 0.225, 46)
    for k, (lab, c) in enumerate([("smallest", c1), ("middle", c2), ("largest", c3)]):
        ax[0].hist(E[:, k] * 100, bins=bins * 100, histtype="step", lw=2, color=c, label=f"{lab} extent")
    ax[0].axvline(6.5, color=ink2, lw=1, ls="--")
    ax[0].axvline(22, color=ink2, lw=1, ls="--")
    ax[0].set(title="Bounding-box extents after scaling", xlabel="extent [cm]", ylabel="# objects")
    ax[0].legend(frameon=False, loc="upper center")

    ax[1].hist(M * 1000, bins=np.linspace(0, 500, 51), color=c1, edgecolor="#fcfcfb", lw=0.8)
    ax[1].set(title="Mass (350 kg/m³ × hull volume, clipped)", xlabel="mass [g]", ylabel="# objects")
    ax[1].text(0.97, 0.95, f"median {np.median(M) * 1000:.0f} g\nclipped low {np.mean(M <= 0.030001) * 100:.0f}%"
               f"\nclipped high {np.mean(M >= 0.499999) * 100:.0f}%", transform=ax[1].transAxes,
               ha="right", va="top", color=ink)  # fmt: skip

    ax[2].hist(S, bins=np.linspace(0, 1.0, 41), color=c1, edgecolor="#fcfcfb", lw=0.8)
    ax[2].set(title="Applied scale factor", xlabel="scale s", ylabel="# objects")
    ax[2].text(0.03, 0.95, f"scaled (s<1): {np.mean(S < 1 - 1e-9) * 100:.0f}%\nmedian s = {np.median(S):.2f}",
               transform=ax[2].transAxes, ha="left", va="top", color=ink)  # fmt: skip

    lo = min(Eraw[:, 0].min(), E[:, 0].min()) * 100
    ax[3].scatter(Eraw[:, 0] * 100, Eraw[:, 2] * 100, s=9, color=c2, alpha=0.55, lw=0, label="raw")
    ax[3].scatter(E[:, 0] * 100, E[:, 2] * 100, s=9, color=c1, alpha=0.7, lw=0, label="after scaling")
    ax[3].axvline(6.5, color=ink2, lw=1, ls="--")
    ax[3].axhline(22, color=ink2, lw=1, ls="--")
    ax[3].set(xscale="log", yscale="log", title="Smallest vs largest extent", xlabel="smallest extent [cm]",
              ylabel="largest extent [cm]", xlim=(lo * 0.9, None))  # fmt: skip
    ax[3].legend(frameon=False, loc="lower right")

    ax[4].hist(Vol * 1e6, bins=np.logspace(np.log10(max(Vol.min() * 1e6, 1)), np.log10(Vol.max() * 1e6), 40),
               color=c1, edgecolor="#fcfcfb", lw=0.8)  # fmt: skip
    ax[4].set(xscale="log", title="Volume (union of convex hulls)", xlabel="volume [cm³]", ylabel="# objects")

    hk, hv = np.unique(H, return_counts=True)
    ax[5].bar(hk, hv, color=c1, width=0.8)
    ax[5].set(title="Convex pieces per object", xlabel="# collision hulls", ylabel="# objects", yscale="log")
    for k_, v_ in zip(hk, hv):
        ax[5].text(k_, v_ * 1.1, str(v_), ha="center", va="bottom", color=ink, fontsize=8)

    fig.suptitle(f"GSO objects converted for Isaac: {len(ok)} usable of {len(rows)}", fontsize=12,
                 fontweight="bold", color=ink)  # fmt: skip
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return stats


def main():
    args = _parse()
    catalog_path = Path(args.catalog)
    validation = {}
    if Path(args.validation).is_file():
        validation = json.loads(Path(args.validation).read_text())

    if args.stats_only:
        rows = json.loads(catalog_path.read_text())
        rows = gso.apply_validation(rows, validation)
        gso.save_catalog(rows, catalog_path)
        st = make_stats(rows, RESULTS / "figures" / "gso_stats.png", RESULTS / "gso_stats.json")
        print(json.dumps({k: st[k] for k in ("n_total", "n_ok", "not_ok_reasons", "frac_scaled")}, indent=1))
        return

    if args.format == "usd":
        _ensure_pxr(args)
        if not gso.pxr_available():
            sys.exit("pxr not importable even after re-exec; use --format usda")

    names = sorted(d for d in os.listdir(args.raw_dir) if os.path.isdir(os.path.join(args.raw_dir, d)))
    if args.names:
        missing = set(args.names) - set(names)
        if missing:
            sys.exit(f"unknown objects: {sorted(missing)}")
        names = sorted(args.names)
    if args.limit:
        names = names[: args.limit]

    old = {}
    if catalog_path.is_file():
        old = {r["name"]: r for r in json.loads(catalog_path.read_text())}
    if args.skip_existing:
        names = [n for n in names if not (n in old and Path(old[n].get("usd", "")).is_file())]

    cfg = gso.ConvertCfg(fmt=args.format, flip_v=args.flip_v)
    dirs = (args.raw_dir, args.usd_dir, args.points_dir)
    os.makedirs(args.usd_dir, exist_ok=True)
    os.makedirs(args.points_dir, exist_ok=True)
    print(f"[convert_gso] converting {len(names)} objects with {args.workers} workers -> {args.usd_dir}", flush=True)
    t0 = time.time()
    new = []
    with Pool(args.workers, initializer=_init, initargs=(cfg, dirs)) as pool:
        for i, e in enumerate(pool.imap_unordered(_work, names, chunksize=2)):
            new.append(e)
            if not e.get("ok"):
                print(f"  [not ok] {e['name']}: {e.get('notes')}", flush=True)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(names)}  {time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0
    rows = dict(old)
    for e in new:
        rows[e["name"]] = e
    rows = list(rows.values())
    rows = gso.apply_validation(rows, validation)
    gso.save_catalog(rows, catalog_path)
    n_ok = sum(bool(e.get("ok")) for e in new)
    print(f"[convert_gso] done {len(new)} in {dt:.1f}s ({dt / max(len(new), 1):.2f}s/obj wall); ok {n_ok}/{len(new)}")
    print(f"[convert_gso] catalog: {catalog_path} ({len(rows)} rows, {sum(bool(r.get('ok')) for r in rows)} ok)")
    if not args.no_stats:
        st = make_stats(rows, RESULTS / "figures" / "gso_stats.png", RESULTS / "gso_stats.json")
        print(json.dumps({k: st[k] for k in ("n_total", "n_ok", "not_ok_reasons", "frac_scaled")}, indent=1))


if __name__ == "__main__":
    main()
