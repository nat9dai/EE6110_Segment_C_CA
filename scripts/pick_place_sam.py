#!/usr/bin/env python
"""Pick an object chosen by name, slot or x,y and place it in the detected basket, using SAM masks.

Settings come from configs/pick_place.yaml; env vars such as EPISODE_TARGET override them.
    EPISODE_TARGET=Tape ~/isaacsim_6_1/python.sh scripts/pick_place_sam.py
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from isaaclab.app import AppLauncher  # noqa: E402

# device="cpu": otherwise Isaac Lab 3.0 calls torch.cuda.set_device and crashes on CPU-only torch
app = AppLauncher(headless=True, enable_cameras=True, device="cpu").app
print("KIT_STARTED", flush=True)

import imageio.v2 as iio  # noqa: E402

from safegrasp.assets.gso import load_catalog  # noqa: E402
from safegrasp.kinematics.panda import Panda  # noqa: E402
from safegrasp.perception.sam_seg import SamSegmenter, detect_basket, match_target, robot_pixel_mask  # noqa: E402
from safegrasp.perception.types import View  # noqa: E402
from safegrasp.pipeline.methods import get_method  # noqa: E402
from safegrasp.pipeline.trial import TrialCfg, WS_ROI, run_trial  # noqa: E402
from safegrasp.sim.scene import GraspScene, SceneCfg  # noqa: E402
from safegrasp.sim.scenarios import make_scene  # noqa: E402

import yaml  # noqa: E402
CONFIG_PATH = os.environ.get("PICKPLACE_CONFIG", os.path.join(ROOT, "configs", "pick_place.yaml"))
_cfg = {}
if os.path.isfile(CONFIG_PATH):
    with open(CONFIG_PATH) as _f:
        _cfg = yaml.safe_load(_f) or {}
    print("config:", CONFIG_PATH, flush=True)


def _get(env, key, default):
    v = os.environ.get(env)
    return v if v not in (None, "") else _cfg.get(key, default)


SEED = int(_get("EPISODE_SEED", "seed", 11))
KIND = str(_get("EPISODE_KIND", "kind", "medium"))
METHOD = str(_get("EPISODE_METHOD", "method", "ant_nom"))
TARGET_SPEC = str(_get("EPISODE_TARGET", "target", ""))        # "" | name | slot index | "x,y"
PLACE_TOL = float(_get("EPISODE_PLACE_TOL", "place_tol", 0.11))
CKPT = str(_get("SAM_CKPT", "sam_checkpoint", "") or os.path.join(ROOT, "data", "sam_vit_b_01ec64.pth"))

_basket = _get("EPISODE_BASKET", "basket", [0.58, -0.33])      # true position; the run uses the detected one
if isinstance(_basket, str):
    bx, by = [float(x) for x in _basket.split(",")[:2]]
else:
    bx, by = float(_basket[0]), float(_basket[1])
catalog = load_catalog()
scene = GraspScene(SceneCfg(pool=catalog, cameras=("main", "left", "right"), basket_pos=(bx, by)))
panda = Panda()
spec = make_scene(SEED, KIND, scene.pool_extents)
snap = scene.place_scene(spec)
scene.robot_reset(panda.q_home, 0.08)
scene.step_physics(3)
vl, rgbs = scene.capture(("main", "left", "right"), with_rgb=True)
views = {v.name: v for v in vl}
rgb_by = {v.name: r for v, r in zip(vl, rgbs)}
poses = scene.object_poses()
active = list(map(int, np.flatnonzero(scene.active)))


def resolve_target(s):
    """Map EPISODE_TARGET (name, slot index or 'x,y') to an object slot."""
    s = s.strip()
    if not s:   # default: first visible, reachable object
        main = views["main"]
        ids, cnt = np.unique(main.seg[main.seg >= 0], return_counts=True)
        vis = dict(zip(ids.tolist(), cnt.tolist()))
        cand = [a for a in active if vis.get(a, 0) >= 300 and 0.30 <= poses[a, 0] <= 0.80
                and abs(poses[a, 1]) <= 0.30 and poses[a, 2] < 0.25]
        return cand[0] if cand else active[0]
    if "," in s:
        xy = np.array([float(x) for x in s.split(",")[:2]])
        return min(active, key=lambda a: float(np.linalg.norm(poses[a, :2] - xy)))
    try:
        return int(s) if int(s) in active else active[0]
    except ValueError:
        for a in active:
            if catalog[a].name.lower() == s.lower() or s.lower() in catalog[a].name.lower():
                return a
        return active[0]


gt_slot = resolve_target(TARGET_SPEC)
print("TARGET slot=%d name=%s pos=%s" % (gt_slot, catalog[gt_slot].name, np.round(poses[gt_slot, :3], 3)), flush=True)

print("loading SAM...", flush=True)
t = time.time()
sam = SamSegmenter(CKPT, device="cpu")
main = views["main"]
rmask = robot_pixel_mask(panda, panda.q_home, 0.08, main.K, main.T_wc, *main.depth.shape[:2])
sam_seg = sam.segment(rgb_by["main"], main.depth, main.K, main.T_wc, robot_mask=rmask, roi=WS_ROI)
print("SAM done %.1fs instances=%d" % (time.time() - t, len(np.unique(sam_seg[sam_seg >= 0]))), flush=True)

# give the target's SAM mask its GT slot id for scoring; SAM's mask errors are kept
sam_inst = match_target(sam_seg, main.depth, main.K, main.T_wc, poses[gt_slot, :2])
seg2 = np.where(sam_seg >= 0, sam_seg + 1000, sam_seg).astype(np.int32)
if sam_inst is not None:
    seg2[sam_seg == sam_inst] = gt_slot
    print("matched target to SAM instance %d (%d px)" % (sam_inst, int((sam_seg == sam_inst).sum())), flush=True)
else:
    print("WARN: target object not found among SAM masks -- grasp will likely fail", flush=True)
main_sam = View(depth=main.depth, K=main.K, T_wc=main.T_wc, seg=seg2, name="main")
views_all = {"main": main_sam, "left": views["left"], "right": views["right"]}

det = detect_basket(rgb_by["main"], main.depth, main.K, main.T_wc)
if det is None:
    print("ERROR: basket not detected in the camera view", flush=True)
    os._exit(0)
bcenter, rim_z, n_px = det
print("BASKET detected at %s rim_z=%.3f (%d px)  [physical %.2f,%.2f -> err %.3f m]" %
      (np.round(bcenter, 3), rim_z, n_px, bx, by, float(np.linalg.norm(bcenter - [bx, by]))), flush=True)
drop = (float(bcenter[0]), float(bcenter[1]), rim_z + 0.10)   # release 10 cm above the detected rim

cfg = TrialCfg(place_pos=drop, place_tol=PLACE_TOL, regrasp_on_empty=True, film_every=4, film_cam="main")
r = run_trial(scene, snap, gt_slot, get_method(METHOD), panda, views_all, noise_seed=SEED, cfg=cfg)
fr = r.pop("_frames", None) or []
print("RESULT success=%s fail=%s phase=%s place_dist=%s frames=%d" %
      (r.get("success"), r.get("fail"), r.get("phase_reached"), r.get("place_dist"), len(fr)), flush=True)
if fr:
    name = catalog[gt_slot].name[:16].replace("/", "_")
    out = os.path.join(ROOT, "results", "pickplace_%s_%d.gif" % (name, SEED))
    iio.mimsave(out, [np.asarray(f[2])[..., :3].astype("uint8") for f in fr], duration=1 / 15.0)
    print("SAVED", out, len(fr), "frames", flush=True)
try:
    app.close()
except Exception:
    pass
os._exit(0)
