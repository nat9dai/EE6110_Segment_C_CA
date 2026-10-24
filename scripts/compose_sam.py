import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True, device="cpu").app
print("KIT_STARTED", flush=True)

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import imageio.v2 as iio
from PIL import Image, ImageDraw
from safegrasp.assets.gso import load_catalog
from safegrasp.kinematics.panda import Panda
from safegrasp.pipeline.methods import get_method
from safegrasp.pipeline.trial import perceive, run_trial, TrialCfg, WS_ROI
from safegrasp.perception.voxel import OCC, UNK
from safegrasp.perception.types import View
from safegrasp.perception.sam_seg import SamSegmenter, detect_basket, match_target, robot_pixel_mask
from safegrasp.sim.scene import GraspScene, SceneCfg
from safegrasp.sim.scenarios import make_scene

SEED = int(os.environ.get("EPISODE_SEED", "11"))
KIND = os.environ.get("EPISODE_KIND", "medium")
METHOD = os.environ.get("EPISODE_METHOD", "ant_nom")
TARGET_SPEC = os.environ.get("EPISODE_TARGET", "")
BASKET = os.environ.get("EPISODE_BASKET", "0.58,-0.33")
CKPT = os.environ.get("SAM_CKPT", os.path.join(ROOT, "data", "sam_vit_b_01ec64.pth"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(ROOT, "results", "figures"))
W, H = 320, 240
C_TARGET, C_OBS, C_UNK = (0.165, 0.471, 0.839), (0.58, 0.58, 0.58), (0.910, 0.482, 0.643)

bx, by = [float(x) for x in BASKET.split(",")[:2]]
catalog = load_catalog()
scene = GraspScene(SceneCfg(pool=catalog, cameras=("main", "left", "right"), basket_pos=(bx, by)))
panda = Panda()
spec = make_scene(SEED, KIND, scene.pool_extents); snap = scene.place_scene(spec)
scene.robot_reset(panda.q_home, 0.08); scene.step_physics(3)
vl, rgbs = scene.capture(("main", "left", "right"), with_rgb=True)
views = {v.name: v for v in vl}; rgb_by = {v.name: r for v, r in zip(vl, rgbs)}
main = views["main"]; poses = scene.object_poses(); active = list(map(int, np.flatnonzero(scene.active)))


def resolve_target(s):
    s = s.strip()
    if not s:
        ids, cnt = np.unique(main.seg[main.seg >= 0], return_counts=True); vis = dict(zip(ids.tolist(), cnt.tolist()))
        cand = [a for a in active if vis.get(a, 0) >= 300 and 0.30 <= poses[a, 0] <= 0.80 and abs(poses[a, 1]) <= 0.30 and poses[a, 2] < 0.25]
        return cand[0] if cand else active[0]
    if "," in s:
        xy = np.array([float(x) for x in s.split(",")[:2]]); return min(active, key=lambda a: float(np.linalg.norm(poses[a, :2] - xy)))
    try:
        return int(s) if int(s) in active else active[0]
    except ValueError:
        for a in active:
            if s.lower() in catalog[a].name.lower():
                return a
        return active[0]


gt_slot = resolve_target(TARGET_SPEC)
print("TARGET", gt_slot, catalog[gt_slot].name, flush=True)
det = detect_basket(rgb_by["main"], main.depth, main.K, main.T_wc); bcenter, rim_z, _ = det
print("BASKET", np.round(bcenter, 3), flush=True)

print("SAM...", flush=True); t = time.time()
sam = SamSegmenter(CKPT, device="cpu")
rmask = robot_pixel_mask(panda, panda.q_home, 0.08, main.K, main.T_wc, *main.depth.shape[:2])
sam_seg = sam.segment(rgb_by["main"], main.depth, main.K, main.T_wc, robot_mask=rmask, roi=WS_ROI)
sam_inst = match_target(sam_seg, main.depth, main.K, main.T_wc, poses[gt_slot, :2])
print("SAM %.1fs instances=%d target_inst=%s" % (time.time() - t, len(np.unique(sam_seg[sam_seg >= 0])), sam_inst), flush=True)
seg2 = np.where(sam_seg >= 0, sam_seg + 1000, sam_seg).astype(np.int32)
if sam_inst is not None:
    seg2[sam_seg == sam_inst] = gt_slot
main_sam = View(depth=main.depth, K=main.K, T_wc=main.T_wc, seg=seg2, name="main")
views_all = {"main": main_sam, "left": views["left"], "right": views["right"]}

rng = np.random.default_rng(1)
sam_img = np.full(sam_seg.shape + (3,), 0.95)
for i in np.unique(sam_seg[sam_seg >= 0]):
    sam_img[sam_seg == i] = rng.uniform(0.35, 0.85, 3)
if sam_inst is not None:
    sam_img[sam_seg == sam_inst] = C_TARGET
sam_img[sam_seg == -2] = (0.55, 0.55, 0.55)
sam_img = (sam_img * 255).astype(np.uint8)

d = main.depth.astype(np.float64); v = np.isfinite(d) & (d > 0)
dn = np.zeros_like(d)
if v.any():
    lo, hi = np.percentile(d[v], [2, 98]); dn[v] = np.clip((d[v] - lo) / (hi - lo + 1e-6), 0, 1)
depth_img = (cm.get_cmap("viridis")(dn)[..., :3] * 255).astype(np.uint8); depth_img[~v] = (25, 25, 30)

per = perceive([main_sam], gt_slot, panda, panda.q_home, get_method(METHOD), np.random.default_rng(SEED), TrialCfg())
labels, inst = per.labels, per.inst
is_occ = labels == OCC; is_unk = labels == UNK
col_t = (is_occ & (inst == gt_slot)).any(2); col_o = is_occ.any(2) & ~col_t; col_u = is_unk.any(2) & ~is_occ.any(2)
occ = np.full(labels.shape[:2] + (3,), 0.96); occ[col_u] = C_UNK; occ[col_o] = C_OBS; occ[col_t] = C_TARGET
fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
ax.imshow(np.transpose(occ, (1, 0, 2)), origin="lower"); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(pad=0.2); fig.canvas.draw()
occ_img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)


def panel(arr, label):
    im = Image.fromarray(np.asarray(arr)[..., :3].astype(np.uint8)).resize((W, H))
    dr = ImageDraw.Draw(im); dr.rectangle([0, 0, W - 1, 14], fill=(20, 20, 20)); dr.text((4, 2), label, fill=(255, 255, 255))
    return np.asarray(im)


sam_p = panel(sam_img, "SAM segmentation  (target=blue)")
dep_p = panel(depth_img, "depth")
occ_p = panel(occ_img, "occupancy  (pink=unknown)")

cfg = TrialCfg(place_pos=(float(bcenter[0]), float(bcenter[1]), rim_z + 0.10), place_tol=0.11,
               regrasp_on_empty=True, film_every=4, film_cam="main")
r = run_trial(scene, snap, gt_slot, get_method(METHOD), panda, views_all, noise_seed=SEED, cfg=cfg)
fr = r.pop("_frames", None) or []
print("RESULT success=%s fail=%s frames=%d" % (r.get("success"), r.get("fail"), len(fr)), flush=True)

frames = []
for f in fr:
    cam = panel(np.asarray(f[2])[..., :3], "robot camera (main view)")
    top = np.concatenate([cam, sam_p], axis=1)
    bot = np.concatenate([dep_p, occ_p], axis=1)
    frames.append(np.concatenate([top, bot], axis=0))
if frames:
    name = catalog[gt_slot].name[:14].replace("/", "_")
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "panels_sam_%s_%d.gif" % (name, SEED))
    iio.mimsave(out, frames, duration=1 / 15.0)
    print("SAVED", out, len(frames), "frames", flush=True)
try: app.close()
except Exception: pass
os._exit(0)
