import os, sys, collections, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True, enable_cameras=True, device="cpu").app
print("KIT_STARTED", flush=True)

from safegrasp.assets.gso import load_catalog
from safegrasp.kinematics.panda import Panda
from safegrasp.pipeline.methods import get_method
from safegrasp.pipeline.trial import run_trial, TrialCfg, WS_ROI
from safegrasp.perception.types import View
from safegrasp.perception.sam_seg import SamSegmenter, detect_basket, match_target, robot_pixel_mask
from safegrasp.sim.scene import GraspScene, SceneCfg
from safegrasp.sim.scenarios import make_scene

KIND = os.environ.get("EPISODE_KIND", "medium")
SEEDS = [int(x) for x in os.environ.get("EPISODE_SEEDS", "1,2,3,4,5,6,7,8,9,10").split(",")]
BASKET = os.environ.get("EPISODE_BASKET", "0.58,-0.33")
CKPT = os.environ.get("SAM_CKPT", os.path.join(ROOT, "data", "sam_vit_b_01ec64.pth"))
bx, by = [float(x) for x in BASKET.split(",")[:2]]

catalog = load_catalog()
scene = GraspScene(SceneCfg(pool=catalog, cameras=("main", "left", "right"), basket_pos=(bx, by)))
panda = Panda()
print("loading SAM once...", flush=True)
sam = SamSegmenter(CKPT, device="cpu")
m = get_method("ant_nom")
print("READY seeds=%d basket=(%.2f,%.2f)" % (len(SEEDS), bx, by), flush=True)

res = collections.Counter()
berr = []
for seed in SEEDS:
    spec = make_scene(seed, KIND, scene.pool_extents)
    snap = scene.place_scene(spec)
    scene.robot_reset(panda.q_home, 0.08); scene.step_physics(3)
    vl, rgbs = scene.capture(("main", "left", "right"), with_rgb=True)
    views = {v.name: v for v in vl}; rgb_by = {v.name: r for v, r in zip(vl, rgbs)}
    main = views["main"]; poses = scene.object_poses(); active = list(map(int, np.flatnonzero(scene.active)))
    ids, cnt = np.unique(main.seg[main.seg >= 0], return_counts=True); vis = dict(zip(ids.tolist(), cnt.tolist()))
    cand = [a for a in active if vis.get(a, 0) >= 300 and 0.30 <= poses[a, 0] <= 0.80 and abs(poses[a, 1]) <= 0.30 and poses[a, 2] < 0.25]
    if not cand:
        print("SEED %d: no target" % seed, flush=True); res["skip"] += 1; continue
    gt_slot = cand[0]
    det = detect_basket(rgb_by["main"], main.depth, main.K, main.T_wc)
    if det is None:
        print("SEED %d: basket not detected" % seed, flush=True); res["no_basket"] += 1; continue
    bcenter, rim_z, _ = det
    berr.append(float(np.linalg.norm(bcenter - [bx, by])))
    rmask = robot_pixel_mask(panda, panda.q_home, 0.08, main.K, main.T_wc, *main.depth.shape[:2])
    sam_seg = sam.segment(rgb_by["main"], main.depth, main.K, main.T_wc, robot_mask=rmask, roi=WS_ROI)
    sam_inst = match_target(sam_seg, main.depth, main.K, main.T_wc, poses[gt_slot, :2])
    seg2 = np.where(sam_seg >= 0, sam_seg + 1000, sam_seg).astype(np.int32)
    if sam_inst is not None:
        seg2[sam_seg == sam_inst] = gt_slot
    main_sam = View(depth=main.depth, K=main.K, T_wc=main.T_wc, seg=seg2, name="main")
    views_all = {"main": main_sam, "left": views["left"], "right": views["right"]}
    cfg = TrialCfg(place_pos=(float(bcenter[0]), float(bcenter[1]), rim_z + 0.10), place_tol=0.11, regrasp_on_empty=True)
    r = run_trial(scene, snap, gt_slot, m, panda, views_all, noise_seed=seed, cfg=cfg)
    succ = bool(r.get("success")); fail = r.get("fail"); ph = r.get("phase_reached")
    picked = ph in ("transport", "place", "release", "retreat_up")   # grasp held long enough to carry
    res["place_ok" if succ else "place_fail"] += 1
    res["pick_ok" if picked else "pick_fail"] += 1
    if not succ:
        res["fail_" + str(fail)] += 1
    print("SEED %2d tgt=%-22s basket_err=%.3f picked=%s placed=%s fail=%s place_dist=%s" %
          (seed, catalog[gt_slot].name[:22], berr[-1], picked, succ, fail, r.get("place_dist")), flush=True)

n = res["place_ok"] + res["place_fail"]
print("\n=== PICK-AND-PLACE PERFORMANCE (n=%d, %s, SAM seg) ===" % (n, KIND), flush=True)
print("  pick  (grasp+carry): %2d/%2d = %3.0f%%" % (res["pick_ok"], n, 100 * res["pick_ok"] / max(n, 1)), flush=True)
print("  place (into basket): %2d/%2d = %3.0f%%" % (res["place_ok"], n, 100 * res["place_ok"] / max(n, 1)), flush=True)
print("  basket detection err: mean %.3f m  max %.3f m" % (np.mean(berr) if berr else 0, np.max(berr) if berr else 0), flush=True)
print("  failures:", {k: v for k, v in res.items() if k.startswith("fail_")}, flush=True)
try: app.close()
except Exception: pass
os._exit(0)
