"""Compare a baseline grasp config with a variant on the same scenes (paired A/B, pick-and-lift).

Each scene runs both arms from the same snapshot and noise seed; rows go to results/raw/ab_regrasp/.
    ~/isaacsim_6_1/python.sh scripts/ab_regrasp.py --n-scenes 24 --kind medium --mode recover
    python3 scripts/ab_regrasp.py --summary --mode recover
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--n-scenes", type=int, default=24)
ap.add_argument("--kind", default="medium")
ap.add_argument("--method", default="ant_nom")
ap.add_argument("--pool-size", type=int, default=40)
ap.add_argument("--pool-seed", type=int, default=7)
ap.add_argument("--obj-iters", type=int, default=16)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--null", action="store_true",
                help="null control: run the baseline config in BOTH arms to measure the PhysX "
                     "non-determinism floor (how many scenes flip with identical code)")
ap.add_argument("--mode", default="recover", choices=["recover", "softclose"],
                help="what arm B is: 'recover' (empty-close lateral-search retry) or "
                     "'softclose' (ramp the fingers shut to avoid pushing the object out)")
ap.add_argument("--summary", action="store_true", help="print the paired tally of the existing log and exit")
args = ap.parse_args()
out_path = os.path.join(ROOT, "results", "raw", "ab_regrasp", "ab_null.jsonl" if args.null else f"ab_{args.mode}.jsonl")
if args.summary:
    by = {}
    for line in open(out_path):
        r = json.loads(line)
        by.setdefault(r["scene_seed"], {})[r["arm"]] = bool(r.get("success"))
    pairs = [v for v in by.values() if len(v) == 2]
    arms = sorted(pairs[0], key=lambda a: a != "base")
    print(f"{len(pairs)} paired scenes, success " + ", ".join(f"{a} {sum(v[a] for v in pairs)}" for a in arms)
          + f", outcome differs in {sum(len(set(v.values())) > 1 for v in pairs)}")
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

# device="cpu": otherwise Isaac Lab 3.0 calls torch.cuda.set_device and crashes on CPU-only torch
app = AppLauncher(headless=True, enable_cameras=True, device="cpu",
                  kit_args=f"--/physics/numThreads={args.threads} "
                           f"--/plugins/carb.tasking.plugin/threadCount={2 * args.threads}").app

import numpy as np  # noqa: E402

from safegrasp.assets.gso import load_catalog  # noqa: E402
from safegrasp.kinematics.panda import Panda  # noqa: E402
from safegrasp.pipeline.methods import get_method  # noqa: E402
from safegrasp.pipeline.trial import TrialCfg, run_trial  # noqa: E402
from safegrasp.sim.scenarios import make_scene  # noqa: E402
from safegrasp.sim.scene import GraspScene, SceneCfg  # noqa: E402

KIND_ID = {"sparse": 0, "medium": 1, "dense": 2, "hidden": 3}
os.makedirs(os.path.dirname(out_path), exist_ok=True)


def log(*a):
    print("[ab]", *a, flush=True)


done = set()
if os.path.exists(out_path):
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["scene_seed"], r["arm"]))
            except Exception:
                pass
log(f"resuming with {len(done)} finished trials")

catalog = load_catalog()
prng = np.random.default_rng(args.pool_seed * 1000)
pool_size = min(args.pool_size, len(catalog))               # the catalogue may be a small sparse checkout
pool = [catalog[i] for i in prng.choice(len(catalog), pool_size, replace=False)]
scene = GraspScene(SceneCfg(pool=pool, obj_pos_iters=args.obj_iters))
panda = Panda()
method = get_method(args.method)

if args.null:
    ARMS = {"base": TrialCfg(), "base2": TrialCfg()}
elif args.mode == "softclose":
    ARMS = {"base": TrialCfg(), "softclose": TrialCfg(soft_close=True, t_close_soft=1.8)}
else:
    ARMS = {
        "base": TrialCfg(),
        "recover": TrialCfg(regrasp_recover=True, regrasp_max=2, regrasp_depth=0.015, regrasp_recenter=1.0),
    }
A, B = list(ARMS)
log(f"pool ready ({len(pool)} objects), method={method.name}, kind={args.kind}, n={args.n_scenes}")


def choose_target(spec, views, rng):
    main = views["main"]
    ids, counts = np.unique(main.seg[main.seg >= 0], return_counts=True)
    vis = dict(zip(ids.tolist(), counts.tolist()))
    poses = scene.object_poses()
    ok = []
    for s in np.flatnonzero(scene.active):
        x, y, z = poses[s, :3]
        if vis.get(int(s), 0) >= 300 and 0.30 <= x <= 0.80 and abs(y) <= 0.30 and z < 0.25:
            ok.append(int(s))
    if spec.target_slot is not None:
        return spec.target_slot if spec.target_slot in ok else None
    return int(rng.choice(ok)) if ok else None


tally = {a: {"n": 0, "succ": 0} for a in ARMS}
pairs = []
t_start = time.time()
for i in range(args.n_scenes):
    seed = 100000 * (1 + KIND_ID[args.kind]) + 1000 * args.pool_seed + i
    try:
        for attempt in range(8):
            spec = make_scene(seed + 10_000_000 * attempt, args.kind, scene.pool_extents)
            snap = scene.place_scene(spec)
            vlist = scene.capture(("main", "left", "right"))
            views = {v.name: v for v in vlist}
            hid = spec.meta.get("hidden", [])
            if not hid or max(int((views["main"].seg == h).sum()) for h in hid) <= 150:
                break
        target = choose_target(spec, views, np.random.default_rng(seed))
        drift = scene.null_drift(snap)
        unstable = [int(s_) for s_ in np.flatnonzero(scene.active) if drift[s_] > 0.01]
    except Exception as e:  # noqa: BLE001
        log("scene error", seed, repr(e)); traceback.print_exc(); continue
    if target is None:
        log(f"scene {seed}: no valid target, skipped"); continue

    row = {}
    for arm, cfg in ARMS.items():
        if (seed, arm) in done:
            # already logged; reload its result for the paired tally
            with open(out_path) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("scene_seed") == seed and r.get("arm") == arm:
                        row[arm] = bool(r.get("success")); break
            continue
        t0 = time.time()
        try:
            r = run_trial(scene, snap, target, method, panda, views, noise_seed=seed, cfg=cfg, exclude=unstable)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            r = dict(success=False, error=repr(e), phase_reached="error")
        r.pop("_frames", None); r.pop("_trace", None)
        r.update(arm=arm, scene_seed=seed, kind=args.kind, method=method.name,
                 target_name=pool[target].name, n_objects=int(scene.active.sum()), unstable=unstable)
        with open(out_path, "a") as f:
            f.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        tally[arm]["n"] += 1
        tally[arm]["succ"] += int(bool(r.get("success")))
        row[arm] = bool(r.get("success"))
        log(f"scene {seed} {arm:8s} tgt={pool[target].name[:26]:26s} "
            f"succ={r.get('success')} fail={r.get('fail')} attempts={r.get('attempts')} "
            f"t={time.time() - t0:.1f}s")
    if A in row and B in row:
        pairs.append((row[A], row[B]))

log("=" * 70)
for arm in ARMS:
    n, s = tally[arm]["n"], tally[arm]["succ"]
    log(f"{arm:8s}: {s}/{n} success" + (f" = {100 * s / n:.0f}%" if n else ""))
if pairs:
    a2b = sum(1 for a, b in pairs if (not a) and b)
    b2a = sum(1 for a, b in pairs if a and (not b))
    log(f"paired (n={len(pairs)}): {B} vs {A} flipped {a2b} failures->success, {b2a} successes->failure "
        f"(net {a2b - b2a:+d}); total discordant pairs = {a2b + b2a}")
log(f"done in {time.time() - t_start:.0f}s")
sys.stdout.flush()
os._exit(0)
