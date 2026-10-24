"""Benchmark worker: run every method on the same restored scenes and append one JSON line per trial.

    ~/isaacsim/python.sh scripts/run_benchmark.py --exp main --worker 0 --n-workers 4 --n-scenes 40
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
ap.add_argument("--exp", default="main")
ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--n-workers", type=int, default=1)
ap.add_argument("--kinds", default="sparse,medium,dense,hidden")
ap.add_argument("--n-scenes", type=int, default=10, help="scenes per kind (split across workers)")
ap.add_argument("--scene-offset", type=int, default=0)
ap.add_argument("--methods", default="td_nom,ant_nom,ant_apf,ant_cbf,ant_oacbf,safe_nom,safe_oacbf")
ap.add_argument("--pool-size", type=int, default=40)
ap.add_argument("--pool-seed", type=int, default=7)
ap.add_argument("--save-rgb", type=int, default=2, help="save main-view RGB for the first N scenes")
ap.add_argument("--trace", default="", help="comma list of methods for which to save step traces")
ap.add_argument("--max-scenes", type=int, default=10**9)
ap.add_argument("--obj-iters", type=int, default=16, help="PhysX solver position iterations for objects")
ap.add_argument("--threads", type=int, default=2, help="PhysX CPU / Kit tasking threads per process")
args = ap.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=True,
                  kit_args=f"--/physics/numThreads={args.threads} --/plugins/carb.tasking.plugin/threadCount={2 * args.threads}").app

import numpy as np  # noqa: E402

from safegrasp.assets.gso import load_catalog  # noqa: E402
from safegrasp.kinematics.panda import Panda  # noqa: E402
from safegrasp.pipeline.methods import get_method  # noqa: E402
from safegrasp.pipeline.trial import TrialCfg, run_trial  # noqa: E402
from safegrasp.sim.scenarios import make_scene  # noqa: E402
from safegrasp.sim.scene import GraspScene, SceneCfg  # noqa: E402

KIND_ID = {"sparse": 0, "medium": 1, "dense": 2, "hidden": 3}
out_dir = os.path.join(ROOT, "results", "raw", args.exp)
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"w{args.worker:02d}.jsonl")
trace_dir = os.path.join(out_dir, "traces")
rgb_dir = os.path.join(out_dir, "rgb")
os.makedirs(trace_dir, exist_ok=True)
os.makedirs(rgb_dir, exist_ok=True)


def log(*a):
    print(f"[w{args.worker}]", *a, flush=True)


done = set()
if os.path.exists(out_path):
    with open(out_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["scene_seed"], r["method"]))
            except Exception:
                pass
log(f"resuming with {len(done)} finished trials")

# each worker draws its own object subset, so together they cover the dataset
catalog = load_catalog()
prng = np.random.default_rng(args.pool_seed * 1000 + args.worker)
pool = [catalog[i] for i in prng.choice(len(catalog), args.pool_size, replace=False)]
scene = GraspScene(SceneCfg(pool=pool, obj_pos_iters=args.obj_iters))
panda = Panda()
methods = [get_method(m) for m in args.methods.split(",")]
trace_methods = set(args.trace.split(",")) if args.trace else set()
cfg = TrialCfg()
log(f"pool ready ({len(pool)} objects), methods={[m.name for m in methods]}")


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


kinds = args.kinds.split(",")
n_done_scenes = 0
t_start = time.time()
for i in range(args.scene_offset, args.scene_offset + args.n_scenes):
    if i % args.n_workers != args.worker:
        continue
    for kind in kinds:
        seed = 100000 * (1 + KIND_ID[kind]) + 1000 * args.pool_seed + i
        todo = [m for m in methods if (seed, m.name) not in done]
        if not todo:
            continue
        if n_done_scenes >= args.max_scenes:
            break
        try:
            for attempt in range(8):  # re-roll until the hidden objects are really hidden
                spec = make_scene(seed + 10_000_000 * attempt, kind, scene.pool_extents)
                snap = scene.place_scene(spec)
                vlist, rgbs = scene.capture(("main", "left", "right"), with_rgb=True)
                views = {v.name: v for v in vlist}
                hid = spec.meta.get("hidden", [])
                if not hid or max(int((views["main"].seg == h).sum()) for h in hid) <= 150:
                    break
            target = choose_target(spec, views, np.random.default_rng(seed))
            drift = scene.null_drift(snap)
            unstable = [int(s_) for s_ in np.flatnonzero(scene.active) if drift[s_] > 0.01]
        except Exception as e:  # noqa: BLE001
            log("scene error", seed, repr(e))
            traceback.print_exc()
            continue
        if target is None:
            log(f"scene {seed} ({kind}): no valid target, skipped")
            for m in todo:
                with open(out_path, "a") as f:
                    f.write(json.dumps(dict(scene_seed=seed, method=m.name, kind=kind, skipped=True)) + "\n")
            continue
        if n_done_scenes < args.save_rgb:
            import imageio.v2 as iio
            for v, rgb in zip(vlist, rgbs):
                iio.imwrite(os.path.join(rgb_dir, f"{seed}_{v.name}.png"), rgb)
        n_done_scenes += 1
        scene_info = dict(
            scene_seed=seed, gen_seed=spec.seed, kind=kind, worker=args.worker, pool_seed=args.pool_seed,
            n_objects=int(scene.active.sum()), target_name=pool[target].name,
            target_extents=[float(x) for x in pool[target].extents], target_mass=float(pool[target].mass),
            objects=[pool[s].name for s in np.flatnonzero(scene.active)],
            hidden=[int(h) for h in spec.meta.get("hidden", [])],
            hidden_px_main=[int((views["main"].seg == h).sum()) for h in spec.meta.get("hidden", [])],
            hidden_px_all=[int(sum((v.seg == h).sum() for v in vlist)) for h in spec.meta.get("hidden", [])],
            unstable=unstable,
        )
        for m in todo:
            cfg.trace = m.name in trace_methods
            t0 = time.time()
            try:
                r = run_trial(scene, snap, target, m, panda, views, noise_seed=seed, cfg=cfg, exclude=unstable)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                r = dict(method=m.name, success=False, error=repr(e), phase_reached="error")
            r.pop("_frames", None)
            tr = r.pop("_trace", None)
            if tr is not None:
                np.save(os.path.join(trace_dir, f"{seed}_{m.name}.npy"), tr)
            r.update(scene_info)
            r["method"] = m.name
            with open(out_path, "a") as f:
                f.write(json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
            log(f"scene {seed} {kind:6s} tgt={pool[target].name[:28]:28s} {m.name:18s} "
                f"succ={r.get('success')} fail={r.get('fail')} coll={r.get('collision')} "
                f"dist={r.get('n_disturbed')} t={time.time() - t0:.1f}s")
log(f"finished {n_done_scenes} scenes in {time.time() - t_start:.0f}s")
sys.stdout.flush()
os._exit(0)
