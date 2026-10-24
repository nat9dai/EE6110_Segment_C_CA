"""Draw the qualitative report figures into results/figures/ from the output of film_episodes.py.

    ~/isaacsim/python.sh scripts/make_figures.py --seed 407000 --methods ant_nom,safe_oacbf
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from safegrasp.kinematics.panda import Panda  # noqa: E402
from safegrasp.perception.pointcloud import backproject_views  # noqa: E402
from safegrasp.perception.shadow import shadow_mask  # noqa: E402
from safegrasp.perception.types import View  # noqa: E402
from safegrasp.perception.voxel import (OCC, VoxelGrid, build_occupancy, unknown_mask_for_target)  # noqa: E402
from safegrasp.pipeline.trial import WS_ROI, target_hull_mask  # noqa: E402

FIG = os.path.join(ROOT, "results", "figures")
FILM = os.path.join(ROOT, "results", "film")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
C_TARGET, C_OBS, C_UNK, C_HID, C_OURS, C_BASE = "#2a78d6", "#8a8984", "#e87ba4", "#e34948", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10,
                     "axes.titlecolor": INK, "axes.titlelocation": "left"})
import plot_style  # noqa: E402

plot_style.apply()


def load_views(seed):
    d = np.load(os.path.join(FILM, f"{seed}_views.npz"))
    views = [View(depth=d[f"{n}_depth"], K=d[f"{n}_K"], T_wc=d[f"{n}_T_wc"], seg=d[f"{n}_seg"], name=n) for n in ("main", "left", "right")]
    rgbs = {n: d[f"{n}_rgb"] for n in ("main", "left", "right")}
    return views, rgbs, d


def barrier_sets(views, target, panda):
    grid = VoxelGrid()
    c, r, _, _ = panda.spheres(panda.q_home, 0.08)
    labels, inst = build_occupancy(views, grid, robot_spheres=(c, r))
    pts, ids, _ = backproject_views(views)
    tpts = pts[ids == target]
    u = unknown_mask_for_target(labels, inst, target, grid, carve_radius=0.02, roi=WS_ROI)
    u &= shadow_mask(views, grid)
    u &= ~target_hull_mask(tpts, grid)
    occ = (labels == OCC) & (inst != target)
    tgt = (labels == OCC) & (inst == target)
    return grid, occ, tgt, u, labels


def fig_perception(seed, target, hidden_pts=None):
    panda = Panda()
    views, rgbs, d = load_views(seed)
    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.45], hspace=0.12, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(rgbs["main"])
    ax.set_title("Main camera (RGB)")
    ax.axis("off")
    ax = fig.add_subplot(gs[0, 2:4])
    dep = views[0].depth.copy()
    dep[dep <= 0] = np.nan
    im = ax.imshow(dep, cmap="Blues_r")
    ax.set_title("Depth [m]")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    ax = fig.add_subplot(gs[0, 4:6])
    seg = views[0].seg
    segv = np.full(seg.shape + (3,), 0.93)
    rng = np.random.default_rng(3)
    for s_ in np.unique(seg[seg >= 0]).astype(int):
        segv[seg == s_] = rng.uniform(0.45, 0.8, 3)
    segv[seg == target] = matplotlib.colors.to_rgb(C_TARGET)
    segv[seg == -2] = 0.62
    ax.imshow(segv)
    ax.set_title("Instance masks (target blue, robot grey)")
    ax.axis("off")
    for k, (vs, title) in enumerate([(views[:1], "Main camera only"), (views, "Main + two side cameras")]):
        grid, occ, tgt, u, _ = barrier_sets(vs, target, panda)
        C = grid.centers(flat=True).reshape(-1, 3)
        ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3], projection="3d")
        for mask, col, a_, lab, sz in [(u.ravel(), C_UNK, 0.28, f"occlusion-shadow barrier set  ({u.sum():,} voxels)", 6),
                                       (occ.ravel(), C_OBS, 0.55, "observed obstacles", 4),
                                       (tgt.ravel(), C_TARGET, 0.95, "target", 5)]:
            P = C[mask]
            if len(P) > 9000:
                P = P[np.random.default_rng(0).choice(len(P), 9000, replace=False)]
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=sz, c=col, alpha=a_, label=lab, depthshade=False, linewidths=0)
        if hidden_pts is not None and len(hidden_pts):
            ax.scatter(hidden_pts[:, 0], hidden_pts[:, 1], hidden_pts[:, 2], s=4, c=C_HID, alpha=0.9, label="hidden object (ground truth)", linewidths=0)
        tc = C[tgt.ravel()].mean(0) if tgt.any() else np.array([0.55, 0.0, 0.05])
        ax.set_xlim(tc[0] - 0.2, tc[0] + 0.2)
        ax.set_ylim(tc[1] - 0.2, tc[1] + 0.2)
        ax.set_zlim(0, 0.25)
        ax.set_box_aspect((1, 1, 0.62))
        ax.view_init(elev=24, azim=-155)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("x", fontsize=7)
        ax.set_ylabel("y", fontsize=7)
        leg = ax.legend(fontsize=7, loc="upper left", frameon=False, markerscale=3)
        for h in leg.legend_handles:
            h.set_alpha(1)
    out = os.path.join(FIG, f"perception_{seed}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_spheres():
    panda = Panda()
    q = panda.q_home.copy()
    c, r, _, ids = panda.spheres(q, 0.08)
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    groups = panda.sphere_groups
    col = np.full(len(r), C_OBS, dtype=object)
    col[groups["hand"]] = C_BASE
    col[groups["fingers"]] = C_TARGET
    for ax, (i, j, lab) in zip(axs, [(0, 2, "side view (x-z)"), (1, 2, "front view (y-z)")]):
        for k in range(len(r)):
            ax.add_patch(Circle((c[k, i], c[k, j]), r[k], facecolor=col[k], alpha=0.35, edgecolor=col[k], lw=0.8))
        T = panda.fk_all(q)
        chain = np.array([T[n][:3, 3] for n in ["panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4",
                                                 "panda_link5", "panda_link6", "panda_link7", "panda_hand"]])
        ax.plot(chain[:, i], chain[:, j], "-o", color=INK, lw=1.2, ms=2.5)
        tcp = panda.tcp(q)[:3, 3]
        ax.plot(tcp[i], tcp[j], "x", color=C_HID, ms=7, mew=2)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_aspect("equal")
        ax.set_title(lab)
        ax.set_xlabel(["x", "y", "z"][i] + " [m]")
        ax.set_ylabel("z [m]")
    axs[0].text(0.02, 0.95, f"{len(r)} spheres: arm (grey), hand (orange), fingers (blue); x = TCP",
                transform=axs[0].transAxes, fontsize=7.5, color=INK2, va="top")
    fig.tight_layout()
    out = os.path.join(FIG, "panda_spheres.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


PH_COL = {0: "#f4f3ef", 1: "#e9f1fb", 2: "#fdf0e8", 3: "#eef7f2", 4: "#f7f0f5", 5: "#f4f3ef"}
PH_NAME = ["transit", "approach", "close", "lift", "hold", "retreat"]


def fig_traces(seed, methods):
    fig, axs = plt.subplots(len(methods), 1, figsize=(8, 2.4 * len(methods)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, m in zip(axs, methods):
        p = os.path.join(FILM, f"{seed}_{m}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        tr = d["trace"]
        rec = json.loads(str(d["record"]))
        if tr.ndim != 2 or len(tr) == 0:
            continue
        t, ph = tr[:, 0], tr[:, 1].astype(int)
        start = 0
        for k in range(1, len(t) + 1):
            if k == len(t) or ph[k] != ph[start]:
                ax.axvspan(t[start], t[k - 1], color=PH_COL.get(ph[start], "#fff"), lw=0)
                ax.text(0.5 * (t[start] + t[k - 1]), 0.118, PH_NAME[ph[start]], ha="center", fontsize=6.5, color=MUTED)
                start = k
        cols = tr.shape[1]
        if cols >= 16:
            ax.plot(t, tr[:, 15], color=INK, lw=1.4, label="true clearance to non-target objects (GT)")
        ax.plot(t, tr[:, 13], color=C_OBS, lw=1.2, label="h_obs (observed obstacles)")
        if not np.all(np.isnan(tr[:, 14])):
            ax.plot(t, tr[:, 14], color=C_UNK, lw=1.2, label="h_unk (occlusion shadow)")
        ax.axhline(0, color=C_HID, lw=0.8, ls="--")
        if cols >= 17:
            f = tr[:, 16]
            hit = f > 1.0
            if hit.any():
                ax.scatter(t[hit], np.full(hit.sum(), -0.01), s=6, color=C_HID, label="robot touches a non-target object", zorder=5)
        ax.set_ylim(-0.02, 0.125)
        ax.set_ylabel("distance [m]")
        ok = "success" if rec.get("success") else f"fail: {rec.get('fail')}"
        ax.set_title(f"{m}  ({ok}; collision={rec.get('collision')}, disturbed={rec.get('n_disturbed')})")
        ax.legend(fontsize=6.5, loc="upper right", frameon=False, ncol=2)
    axs[-1].set_xlabel("time [s]")
    fig.tight_layout()
    out = os.path.join(FIG, f"traces_{seed}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_traces_bench(seed, methods, exp="main"):
    """Plot the barrier and clearance traces saved by the benchmark run."""
    import glob
    recs = {}
    for p in glob.glob(os.path.join(ROOT, "results", "raw", exp, "w*.jsonl")):
        for line in open(p):
            r = json.loads(line)
            if r.get("scene_seed") == seed and r.get("method") in methods:
                recs[r["method"]] = r
    fig, axs = plt.subplots(len(methods), 1, figsize=(8.4, 2.5 * len(methods)), sharex=True)
    axs = np.atleast_1d(axs)
    for ax, m in zip(axs, methods):
        p = os.path.join(ROOT, "results", "raw", exp, "traces", f"{seed}_{m}.npy")
        if not os.path.exists(p):
            continue
        tr = np.load(p)
        rec = recs.get(m, {})
        t, ph = tr[:, 0], tr[:, 1].astype(int)
        start = 0
        for k in range(1, len(t) + 1):
            if k == len(t) or ph[k] != ph[start]:
                ax.axvspan(t[start], t[k - 1], color=PH_COL.get(ph[start], "#fff"), lw=0)
                ax.text(0.5 * (t[start] + t[k - 1]), 0.128, PH_NAME[ph[start]], ha="center", fontsize=8, color=MUTED)
                start = k
        ax.plot(t, np.clip(tr[:, 15], -0.03, 0.14), color=INK, lw=1.5, label="true clearance, robot ↔ other objects (GT)")
        if not np.all(np.isnan(tr[:, 13])):
            ax.plot(t, np.clip(tr[:, 13], -0.03, 0.14), color=C_OBS, lw=1.2, label="h_obs (CBF, observed obstacles)")
        if not np.all(np.isnan(tr[:, 14])):
            ax.plot(t, np.clip(tr[:, 14], -0.03, 0.14), color=C_UNK, lw=1.2, label="h_unk (CBF, occlusion shadow)")
        hit = tr[:, 16] > 1.0
        if hit.any():
            ax.scatter(t[hit], np.full(hit.sum(), -0.022), s=7, color=C_HID, label="contact with a non-target object (> 1 N)", zorder=5)
        ax.axhline(0, color=C_HID, lw=0.8, ls="--")
        ax.set_ylim(-0.03, 0.14)
        ax.set_ylabel("distance [m]")
        ok = "success" if rec.get("success") else f"failed ({rec.get('fail')})"
        ax.set_title(f"{m}: {ok}, collision = {rec.get('collision')}", fontsize=10.5)
    # one shared legend above the panels keeps the phase labels readable
    handles = {}
    for ax in axs:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    axs[-1].set_xlabel("time [s]")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = os.path.join(FIG, f"traces_bench_{seed}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


ABL_DESC = {
    "safe_oacbf": "reference: safety-ranked + OA-CBF, main camera only",
    "safe_oacbf@views=main+left+right": "+ two side cameras (fewer occlusion shadows)",
    "ant_nom@views=main+left+right": "baseline with three cameras",
    "ant_cbf@views=main+left+right": "observed-only CBF with three cameras",
    "safe_oacbf@alpha=1": "CBF gain α = 1 s⁻¹ (very cautious)",
    "safe_oacbf@alpha=2": "α = 2 s⁻¹",
    "safe_oacbf@alpha=15": "α = 15 s⁻¹ (aggressive)",
    "safe_oacbf@margin_unk=0.0": "no margin to the shadow set",
    "safe_oacbf@margin_unk=0.015": "1.5 cm margin to the shadow set",
    "safe_oacbf@margin_obs=0.005": "0.5 cm margin to observed obstacles",
    "safe_oacbf@margin_obs=0.02": "2 cm margin to observed obstacles",
    "safe_oacbf@carve_radius=0.0": "no target-shadow carving (hull only)",
    "safe_oacbf@carve_radius=0.04": "4 cm target-shadow carving",
    "safe_oacbf@unk_soft=false": "hard (no-slack) shadow barrier",
    "safe_oacbf@noise=0": "perfect depth",
    "safe_oacbf@noise=3": "3× depth noise",
    "ant_nom@noise=0": "baseline, perfect depth",
    "ant_nom@noise=3": "baseline, 3× depth noise",
}


def fig_ablation(exp="abl"):
    """Dot plot of success, deadlock and collision for each ablation variant."""
    d = json.load(open(os.path.join(ROOT, "results", f"summary_{exp}.json")))
    order = [m for m in ABL_DESC if m in d["by_method"]]
    desc = ABL_DESC
    metrics = [("success", "success"), ("deadlock", "deadlock"), ("collision", "collision")]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 0.34 * len(order) + 1.2), sharey=True)
    ys = np.arange(len(order))[::-1]
    for ax, (key, lab) in zip(axs, metrics):
        ref = d["by_method"]["safe_oacbf"][key][0]
        ax.axvline(100 * ref, color=C_OURS, lw=1, ls="--", alpha=0.7)
        for y, m in zip(ys, order):
            p, lo, hi = d["by_method"][m][key]
            col = C_OURS if m == "safe_oacbf" else (C_BASE if m.startswith("ant_") else INK2)
            ax.plot([100 * lo, 100 * hi], [y, y], color=col, lw=1.2, alpha=0.6)
            ax.plot(100 * p, y, "o", color=col, ms=5.5, mec="white", mew=0.8)
        ax.set_title(lab, fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of trials")
        ax.grid(axis="x", color=GRID, lw=0.6)
    axs[0].set_yticks(ys)
    # Liberation Serif lacks a superscript minus, so use math text
    axs[0].set_yticklabels([desc.get(m, m).replace("s⁻¹", r"s$^{-1}$") for m in order], fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG, f"{exp}_dots.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_filmstrip(seed, methods, n=5, crop=None):
    rows = []
    for m in methods:
        p = os.path.join(FILM, f"{seed}_{m}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        f, ph = d["frames"], d["phases"]
        if f.ndim != 4 or len(f) == 0:
            continue
        keep = [i for i in range(len(f)) if ph[i] in ("approach", "close", "lift", "hold", "retreat")]
        sel = [keep[int(round(k))] for k in np.linspace(0, len(keep) - 1, n)] if len(keep) >= n else keep
        imgs = [f[i] if crop is None else f[i][crop[0]:crop[1], crop[2]:crop[3]] for i in sel]
        rows.append((m, imgs, [str(ph[i]) for i in sel], json.loads(str(d["record"]))))
    if not rows:
        return None
    fig, axs = plt.subplots(len(rows), n, figsize=(2.2 * n, 1.8 * len(rows)))
    axs = np.atleast_2d(axs)
    for r, (m, imgs, phs, rec) in enumerate(rows):
        for c in range(n):
            ax = axs[r, c]
            ax.axis("off")
            if c < len(imgs):
                ax.imshow(imgs[c])
                ax.set_title(phs[c], fontsize=7, color=INK2)
        ok = "success" if rec.get("success") else rec.get("fail")
        axs[r, 0].text(-0.08, 0.5, f"{m}\n{ok}\ncollision={rec.get('collision')}", transform=axs[r, 0].transAxes,
                       ha="right", va="center", fontsize=7.5, color=INK)
    fig.tight_layout()
    out = os.path.join(FIG, f"film_{seed}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def gif_side_by_side(seed, methods, out_name=None, scale=2, fps=12, max_frames=90):
    """Write a side-by-side video of one scene under several methods."""
    import imageio.v2 as iio
    from PIL import Image, ImageDraw, ImageFont
    seqs, labels = [], []
    for m in methods:
        p = os.path.join(FILM, f"{seed}_{m}.npz")
        if not os.path.exists(p):
            return None
        d = np.load(p)
        f = d["frames"]
        if f.ndim != 4 or len(f) == 0:
            return None
        rec = json.loads(str(d["record"]))
        seqs.append(f[:, ::scale, ::scale])
        ok = "success" if rec.get("success") else str(rec.get("fail"))
        labels.append(f"{m}: {ok}, collision={rec.get('collision')}")
    n = min(max(len(x) for x in seqs), max_frames * 3)
    step = max(1, n // max_frames)
    frames = []
    for i in range(0, n, step):
        tiles = []
        for sq, lab in zip(seqs, labels):
            im = Image.fromarray(sq[min(i, len(sq) - 1)])
            dr = ImageDraw.Draw(im)
            font = ImageFont.truetype(plot_style.ttf_path(), 13) if plot_style.ttf_path() else None
            dr.rectangle([0, 0, im.width, 16], fill=(20, 25, 29))
            dr.text((5, 1), lab, fill=(235, 235, 235), font=font)
            tiles.append(np.asarray(im))
        frames.append(np.concatenate(tiles, 1))
    out = os.path.join(FIG, out_name or f"episode_{seed}.mp4")
    if out.endswith(".mp4"):  # H.264 is ~20x smaller than GIF; even dimensions required
        H, W = frames[0].shape[:2]
        frames = [fr[: H - H % 2, : W - W % 2] for fr in frames]
        iio.mimsave(out, frames, fps=fps, codec="libx264", quality=7, macro_block_size=1,
                    ffmpeg_params=["-pix_fmt", "yuv420p"])
    else:
        iio.mimsave(out, frames, duration=1000 / fps, loop=0)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, action="append", default=[])
    ap.add_argument("--methods", default="ant_nom,safe_oacbf")
    ap.add_argument("--crop", default="")
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    print(fig_spheres())
    for seed in a.seed:
        ms = a.methods.split(",")
        d = np.load(os.path.join(FILM, f"{seed}_{ms[0]}.npz"))
        target = int(d["target"])
        hidden_pts = None
        hp_path = os.path.join(FILM, f"{seed}_hidden_pts.npy")  # for illustration only
        if os.path.exists(hp_path):
            hidden_pts = np.load(hp_path)
        print(fig_perception(seed, target, hidden_pts))
        print(fig_traces(seed, ms))
        crop = tuple(int(x) for x in a.crop.split(",")) if a.crop else None
        print(fig_filmstrip(seed, ms, crop=crop))
        print(gif_side_by_side(seed, ms))
