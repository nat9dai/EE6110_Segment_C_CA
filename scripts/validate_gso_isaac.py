#!/usr/bin/env python
"""Drop converted GSO objects in Isaac Lab and record in the catalog which ones fail to settle.

    ~/isaacsim/python.sh -u scripts/validate_gso_isaac.py --seed 0 --n 40 --log-file LOG > LOG 2>&1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--names", nargs="*", default=None)
ap.add_argument("--all", action="store_true", help="validate the whole catalog in batches")
ap.add_argument("--batch-size", type=int, default=260)
ap.add_argument("--batch-index", type=int, default=0)
ap.add_argument("--device", default="cpu")
ap.add_argument("--spacing", type=float, default=0.35)
ap.add_argument("--z0", type=float, default=0.15)
ap.add_argument("--duration", type=float, default=3.0)
ap.add_argument("--dt", type=float, default=1.0 / 120.0)
ap.add_argument("--render", action="store_true", help="render a textured sample image instead")
ap.add_argument("--image", default=str(REPO / "results" / "figures" / "gso_samples.png"))
ap.add_argument("--log-file", default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--no-write", action="store_true", help="do not update validation/catalog files")
args = ap.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True, enable_cameras=args.render).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import (  # noqa: E402
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
import isaaclab.utils.math as math_utils  # noqa: E402

from safegrasp.assets import gso  # noqa: E402

T_START = time.time()


def log(*a):
    print("[validate_gso]", *a, flush=True)


def quat_to_R(q):  # (w,x,y,z) numpy -> (3,3)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def choose_entries():
    rows = json.loads(gso.CATALOG_PATH.read_text())
    cands = sorted([r for r in rows if r.get("convert_ok", r.get("ok"))], key=lambda r: r["name"])
    if args.render:
        cands = [r for r in cands if r.get("ok")]
    by = {r["name"]: r for r in cands}
    if args.names:
        sel = [by[n] for n in args.names]
    elif args.all:
        sel = cands[args.batch_index * args.batch_size : (args.batch_index + 1) * args.batch_size]
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(cands), size=min(args.n, len(cands)), replace=False)
        sel = [cands[i] for i in sorted(idx)]
    return sel


def make_collection(entries, positions, quats, root="/World/Objects"):
    objs = {}
    for i, (e, p, q) in enumerate(zip(entries, positions, quats)):
        objs[f"obj_{i:03d}"] = RigidObjectCfg(
            prim_path=f"{root}/obj_{i:03d}",
            spawn=sim_utils.UsdFileCfg(usd_path=e["usd"]),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(map(float, p)), rot=tuple(map(float, q))),
        )
    return RigidObjectCollection(RigidObjectCollectionCfg(rigid_objects=objs))


def scan_log(n_obj):
    """Collect Kit errors and warnings from the stdout log, keyed by the object they name."""
    per_obj, general = {}, []
    if not args.log_file or not os.path.isfile(args.log_file):
        return per_obj, general
    sys.stdout.flush()
    pat = re.compile(r"/World/Objects/obj_(\d{3})")
    for line in open(args.log_file, errors="replace"):
        if "[Error]" not in line and "[Warning]" not in line and "PhysX" not in line:
            continue
        if "[validate_gso]" in line:
            continue
        m = pat.search(line)
        if m:
            per_obj.setdefault(int(m.group(1)), []).append(line.strip()[:300])
        elif "[Error]" in line:
            general.append(line.strip()[:300])
    return per_obj, general


def run_sim_check():
    entries = choose_entries()
    n = len(entries)
    tag = args.tag or (f"all_b{args.batch_index}" if args.all else f"seed{args.seed}_n{n}")
    log(f"subset {tag}: {n} objects, device={args.device}")
    side = int(math.ceil(math.sqrt(n)))
    pos0 = np.array(
        [[(i % side - (side - 1) / 2) * args.spacing, (i // side - (side - 1) / 2) * args.spacing, args.z0] for i in range(n)]
    )
    quat0 = np.tile([1.0, 0, 0, 0], (n, 1))

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device=args.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2000.0))
    t0 = time.time()
    coll = make_collection(entries, pos0, quat0)
    # a standalone copy of entry 0 tests the single-object pose API
    single = RigidObject(
        RigidObjectCfg(
            prim_path="/World/Single",
            spawn=sim_utils.UsdFileCfg(usd_path=entries[0]["usd"]),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -(side + 2) * args.spacing, entries[0]["z_rest"] + 0.002)),
        )
    )
    t_spawn = time.time() - t0
    t0 = time.time()
    sim.reset()
    t_reset = time.time() - t0
    log(f"spawn {t_spawn:.2f}s  reset {t_reset:.2f}s  num_objects={coll.num_objects}")
    dev = sim.device
    pts = [np.load(e["points"]).astype(np.float64) for e in entries]

    n_steps = int(round(args.duration / args.dt))
    step_times = []
    max_speed = np.zeros(n)
    pos_hist = {}
    for k in range(n_steps):
        ts = time.perf_counter()
        sim.step(render=False)
        step_times.append(time.perf_counter() - ts)
        coll.update(args.dt)
        single.update(args.dt)
        if k % 6 == 0 or k == n_steps - 1:
            v = coll.data.object_lin_vel_w[0].cpu().numpy()
            max_speed = np.maximum(max_speed, np.nan_to_num(np.linalg.norm(v, axis=1), nan=1e9))
        if k == int(round(2.5 / args.dt)) - 1:
            pos_hist[2.5] = coll.data.object_pos_w[0].cpu().numpy().copy()
    P = coll.data.object_pos_w[0].cpu().numpy()
    Q = coll.data.object_quat_w[0].cpu().numpy()
    V = coll.data.object_lin_vel_w[0].cpu().numpy()
    W = coll.data.object_ang_vel_w[0].cpu().numpy()
    step_ms = 1000 * np.array(step_times[10:])
    log(f"physics step: mean {step_ms.mean():.2f} ms, median {np.median(step_ms):.2f} ms, p95 {np.percentile(step_ms, 95):.2f} ms"
        f" ({n} objects, {sum(e['n_hulls'] for e in entries)} convex shapes)")  # fmt: skip

    per_obj_log, general_err = scan_log(n)
    results = {}
    for i, e in enumerate(entries):
        notes = []
        R = quat_to_R(Q[i])
        wp = pts[i] @ R.T + P[i]
        zb = float(wp[:, 2].min()) if np.all(np.isfinite(wp)) else float("nan")
        disp = float(np.linalg.norm(P[i, :2] - pos0[i, :2]))
        creep = float(np.linalg.norm(P[i] - pos_hist[2.5][i])) if 2.5 in pos_hist else 0.0
        ls, as_ = float(np.linalg.norm(V[i])), float(np.linalg.norm(W[i]))
        tilt = float(np.degrees(np.arccos(np.clip(abs(R[2, 2]), -1, 1))))
        if not np.all(np.isfinite(P[i])) or not np.isfinite(zb):
            notes.append("NaN pose")
        else:
            if zb < -0.01:
                notes.append(f"sinks: bottom z = {zb * 1000:.1f} mm")
            if zb > 0.01:
                notes.append(f"floats/not settled: bottom z = {zb * 1000:.1f} mm")
            if disp > 0.25 or P[i, 2] > 0.5:
                notes.append(f"flies away: xy displacement {disp:.2f} m")
            if ls > 0.03 or as_ > 0.5 or creep > 0.01:
                notes.append(f"not at rest after {args.duration:.0f}s: |v|={ls:.3f} m/s |w|={as_:.2f} rad/s creep={creep * 1000:.0f} mm/0.5s")
            if max_speed[i] > 3.0:
                notes.append(f"violent motion: max |v| = {max_speed[i]:.1f} m/s")
        errs = [l for l in per_obj_log.get(i, []) if "[Error]" in l or "PhysX" in l]
        if errs:
            notes.append(f"load/physx error: {errs[0][:160]}")
        results[e["name"]] = dict(
            ok=len(notes) == 0, notes=notes, z_bottom=zb, disp_xy=disp, lin_speed=ls, ang_speed=as_,
            tilt_deg=tilt, creep=creep, max_speed=float(max_speed[i]), subset=tag, device=args.device,
            n_log_warnings=len(per_obj_log.get(i, [])),
        )  # fmt: skip
    bad = {k: v["notes"] for k, v in results.items() if not v["ok"]}
    log(f"objects failing: {len(bad)}/{n}")
    for k, v in bad.items():
        log(f"  FAIL {k}: {v}")
    zbs = np.array([r["z_bottom"] for r in results.values()])
    log(f"bottom z [mm]: min {np.nanmin(zbs) * 1000:.2f} median {np.nanmedian(zbs) * 1000:.2f} max {np.nanmax(zbs) * 1000:.2f}")

    api = {}
    tgt = torch.tensor([[0.3, -(side + 2) * args.spacing, 0.2, 0.7071068, 0.0, 0.7071068, 0.0]], device=dev)
    single.write_root_pose_to_sim(tgt)
    single.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
    got_before_step = single.data.root_pos_w[0].cpu().numpy().copy()
    sim.step(render=False)
    single.update(args.dt)
    got = single.data.root_pos_w[0].cpu().numpy()
    api["single_write_root_pose_err_before_step_m"] = float(np.linalg.norm(got_before_step - tgt[0, :3].cpu().numpy()))
    api["single_write_root_pose_err_after_1step_m"] = float(np.linalg.norm(got - tgt[0, :3].cpu().numpy()))
    api["single_root_quat_after"] = single.data.root_quat_w[0].cpu().numpy().round(4).tolist()
    ids = torch.tensor([0, 1], device=dev)
    park = torch.tensor([[[50.0, 0.0, 1.0, 1, 0, 0, 0], [50.0, 1.0, 1.0, 1, 0, 0, 0]]], device=dev)
    coll.write_object_pose_to_sim(park, env_ids=torch.tensor([0], device=dev), object_ids=ids)
    coll.write_object_velocity_to_sim(torch.zeros(1, 2, 6, device=dev), env_ids=torch.tensor([0], device=dev), object_ids=ids)
    sim.step(render=False)
    coll.update(args.dt)
    got = coll.data.object_pos_w[0, :2].cpu().numpy()
    api["collection_park_err_m"] = float(np.abs(got - park[0, :, :3].cpu().numpy()).max())
    back = torch.tensor([[[0.5, 0.0, entries[0]["z_rest"] + 0.003, 1, 0, 0, 0]]], device=dev)
    coll.write_object_pose_to_sim(back, env_ids=torch.tensor([0], device=dev), object_ids=ids[:1])
    coll.write_object_velocity_to_sim(torch.zeros(1, 1, 6, device=dev), env_ids=torch.tensor([0], device=dev), object_ids=ids[:1])
    for _ in range(60):
        sim.step(render=False)
        coll.update(args.dt)
    got = coll.data.object_pos_w[0, 0].cpu().numpy()
    api["collection_teleport_back_settled_pos"] = got.round(4).tolist()
    api["collection_teleport_back_xy_err_m"] = float(np.linalg.norm(got[:2] - np.array([0.5, 0.0])))
    log(f"pose API: {json.dumps(api)}")

    summary = dict(
        tag=tag, device=args.device, n_objects=n, n_convex_shapes=int(sum(e["n_hulls"] for e in entries)),
        dt=args.dt, n_steps=n_steps, step_ms_mean=float(step_ms.mean()), step_ms_median=float(np.median(step_ms)),
        step_ms_p95=float(np.percentile(step_ms, 95)), spawn_s=t_spawn, reset_s=t_reset,
        n_fail=len(bad), failures=bad, z_bottom_mm=dict(min=float(np.nanmin(zbs) * 1000),
        median=float(np.nanmedian(zbs) * 1000), max=float(np.nanmax(zbs) * 1000)), pose_api=api,
        general_errors=general_err[:20], n_general_errors=len(general_err),
        names=[e["name"] for e in entries], wall_s=time.time() - T_START,
    )  # fmt: skip
    out = REPO / "results" / "gso_validation" / f"{tag}_{args.device.replace(':', '')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, objects=results), indent=1))
    log(f"wrote {out}")

    if not args.no_write:
        vpath = gso.VALIDATION_PATH
        val = json.loads(vpath.read_text()) if vpath.is_file() else {}
        for k, r in results.items():
            old = val.get(k)
            if old is not None and not old.get("ok", True) and r["ok"]:
                # a failure from another subset stands
                if old.get("subset") != tag:
                    old.setdefault("also_passed_in", []).append(tag)
                    continue
            val[k] = r
        vpath.write_text(json.dumps(val, indent=1))
        rows = json.loads(gso.CATALOG_PATH.read_text())
        rows = gso.apply_validation(rows, val)
        gso.save_catalog(rows, gso.CATALOG_PATH)
        log(f"updated {vpath} ({len(val)} objects) and catalog ({sum(r['ok'] for r in rows)} ok)")


def run_render():
    from isaaclab.sensors import Camera, CameraCfg

    entries = choose_entries()
    n = len(entries)
    ncol = int(math.ceil(math.sqrt(n * 4 / 3)))
    nrow = int(math.ceil(n / ncol))
    sp = 0.26
    rng = np.random.default_rng(args.seed)
    pos0, quat0 = [], []
    for i, e in enumerate(entries):
        x = (i % ncol - (ncol - 1) / 2) * sp
        y = ((nrow - 1) / 2 - i // ncol) * sp
        yaw = rng.uniform(-math.pi, math.pi)
        pos0.append([x, y, e["z_rest"] + 0.004])
        quat0.append([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args.dt, device=args.device))
    gcfg = sim_utils.GroundPlaneCfg(color=(0.42, 0.42, 0.40))
    gcfg.func("/World/ground", gcfg)
    lcfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0))
    lcfg.func("/World/DomeLight", lcfg)
    dcfg = sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0)
    dcfg.func("/World/SunLight", dcfg, orientation=(0.94, 0.2, 0.25, 0.0))
    coll = make_collection(entries, pos0, quat0)
    cam = Camera(
        CameraCfg(
            prim_path="/World/cam", height=768, width=1024,
            data_types=["rgb", "instance_id_segmentation_fast", "distance_to_image_plane"],
            update_latest_camera_pose=True,  # else the camera pose stays at spawn
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 10.0)),
        )  # fmt: skip
    )
    sim.reset()
    dev = sim.device
    extent = max(ncol, nrow * 4 / 3) * sp
    eye = torch.tensor([[0.0, -0.42 * extent, 1.0 * extent + 0.2]], device=dev)
    cam.set_world_poses_from_view(eye, torch.tensor([[0.0, 0.0, 0.0]], device=dev))
    for _ in range(90):  # settle 0.75 s
        sim.step(render=False)
        coll.update(args.dt)
    for k in range(40):  # textures load and annotators lag for a few frames
        sim.step(render=True)
        try:
            cam.update(args.dt)
            _ = cam.data.output["rgb"]
        except RuntimeError as ex:
            log(f"camera frame {k} not ready ({ex}); retrying")
    rgb = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
    seg_info = cam.data.info[0]["instance_id_segmentation_fast"]["idToLabels"]
    from PIL import Image, ImageDraw

    img = Image.fromarray(rgb)
    Path(args.image).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.image)
    log(f"saved {args.image}  rgb mean {rgb.mean():.1f}")
    labels = sorted(set(seg_info.values()))
    log(f"instance_id_segmentation_fast labels ({len(labels)}): {labels[:40]}")
    ann = img.copy()
    dr = ImageDraw.Draw(ann)
    P = coll.data.object_pos_w[0].cpu().numpy()
    K = cam.data.intrinsic_matrices[0].cpu().numpy()
    T = np.eye(4)
    T[:3, :3] = math_utils.matrix_from_quat(cam.data.quat_w_ros[0]).cpu().numpy()
    T[:3, 3] = cam.data.pos_w[0].cpu().numpy()
    Ti = np.linalg.inv(T)
    log(f"cam pos {T[:3, 3].round(3).tolist()}  K diag {np.diag(K).round(1).tolist()}  obj0 {P[0].round(3).tolist()}")
    for e, p in zip(entries, P):
        pc = Ti[:3, :3] @ p + Ti[:3, 3]
        uv = K @ pc
        u, v = uv[0] / uv[2], uv[1] / uv[2]
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        dr.text((u - 60, v + 40), e["name"][:28], fill=(255, 255, 0))
    ann_path = str(Path(args.image).with_name(Path(args.image).stem + "_annotated.png"))
    ann.save(ann_path)
    json.dump(dict(names=[e["name"] for e in entries], seg_labels=labels),
              open(str(Path(args.image).with_suffix(".json")), "w"), indent=1)  # fmt: skip
    log(f"saved {ann_path}")


try:
    if args.render:
        run_render()
    else:
        run_sim_check()
except Exception:
    import traceback

    traceback.print_exc()
finally:
    log(f"total wall {time.time() - T_START:.1f}s")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
