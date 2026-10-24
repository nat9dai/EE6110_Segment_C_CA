"""Film chosen (scene, method) pairs into results/film/ for scripts/make_figures.py.

    ~/isaacsim/python.sh scripts/film_episodes.py --scene hidden:407000 --methods ant_nom,safe_oacbf
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--scene", action="append", required=True, help="kind:seed")
ap.add_argument("--methods", default="ant_nom,ant_cbf,safe_oacbf")
ap.add_argument("--pool-seed", type=int, default=7)
ap.add_argument("--worker", type=int, default=0)
ap.add_argument("--pool-size", type=int, default=40)
ap.add_argument("--every", type=int, default=6)
ap.add_argument("--cam", default="main")
ap.add_argument("--close", action="store_true", help="film with a close-up camera placed beside the target")
args = ap.parse_args()
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True).app
import numpy as np
from safegrasp.assets.gso import load_catalog
from safegrasp.kinematics.panda import Panda
from safegrasp.pipeline.methods import get_method
from safegrasp.pipeline.trial import TrialCfg, run_trial
from safegrasp.sim.scenarios import make_scene
from safegrasp.sim.scene import GraspScene, SceneCfg
out = os.path.join(ROOT, "results", "film"); os.makedirs(out, exist_ok=True)
catalog = load_catalog()
prng = np.random.default_rng(args.pool_seed * 1000 + args.worker)
pool = [catalog[i] for i in prng.choice(len(catalog), args.pool_size, replace=False)]
cams = ("main", "left", "right", "close") if args.close else ("main", "left", "right")
scene = GraspScene(SceneCfg(pool=pool, cameras=cams)); panda = Panda()
for sc in args.scene:
    kind, seed = sc.split(":"); seed = int(seed)
    spec = make_scene(seed, kind, scene.pool_extents)
    snap = scene.place_scene(spec)
    vlist, rgbs = scene.capture(("main", "left", "right"), with_rgb=True)
    views = {v.name: v for v in vlist}
    hid = list(spec.meta.get("hidden", []))
    if hid:
        np.save(os.path.join(out, f"{seed}_hidden_pts.npy"), np.concatenate(scene.object_points_world(hid, snap)))
    np.savez_compressed(os.path.join(out, f"{seed}_views.npz"), **{f"{v.name}_{k}": getattr(v, k) for v in vlist for k in ("depth", "K", "T_wc", "seg")},
                        **{f"{v.name}_rgb": rgb for v, rgb in zip(vlist, rgbs)}, snap=snap, active=scene.active,
                        hidden=np.array(spec.meta.get("hidden", []), int))
    # same target rule as the benchmark
    main = views["main"]; ids, cnt = np.unique(main.seg[main.seg >= 0], return_counts=True); vis = dict(zip(ids.tolist(), cnt.tolist()))
    poses = scene.object_poses()
    ok = [int(s) for s in np.flatnonzero(scene.active) if vis.get(int(s), 0) >= 300 and 0.30 <= poses[s, 0] <= 0.80 and abs(poses[s, 1]) <= 0.30 and poses[s, 2] < 0.25]
    target = spec.target_slot if spec.target_slot is not None else int(np.random.default_rng(seed).choice(ok))
    drift = scene.null_drift(snap)
    unstable = [int(s) for s in np.flatnonzero(scene.active) if drift[s] > 0.01]
    if args.close:
        tp = snap[target, :3]
        scene.look_at("close", (tp[0] + 0.05, tp[1] + 0.30, tp[2] + 0.12), (tp[0], tp[1], tp[2] + 0.02))
    for mname in args.methods.split(","):
        m = get_method(mname)
        cfg = TrialCfg(trace=True, film_every=args.every, film_cam="close" if args.close else args.cam)
        r = run_trial(scene, snap, target, m, panda, views, noise_seed=seed, cfg=cfg, exclude=unstable)
        fr = r.pop("_frames", None) or []
        tr = r.pop("_trace", None)
        np.savez_compressed(os.path.join(out, f"{seed}_{mname}.npz"), frames=np.stack([f[2] for f in fr]) if fr else np.zeros((0,)),
                            times=np.array([f[0] for f in fr]), phases=np.array([f[1] for f in fr]),
                            trace=tr if tr is not None else np.zeros((0,)), target=target,
                            record=json.dumps(r, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        print(f"filmed {seed} {mname}: success={r.get('success')} fail={r.get('fail')} coll={r.get('collision')} frames={len(fr)}", flush=True)
sys.stdout.flush(); os._exit(0)
