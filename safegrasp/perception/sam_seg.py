"""Segment objects with SAM (Segment Anything) in place of the simulator's ground-truth masks.

Masks are kept only if their depth points sit on the table inside the workspace, and the arm is
masked from its own kinematics. Output map: object id >= 0, -1 table or background, -2 robot.
"""
from __future__ import annotations

import numpy as np

from .types import SEG_ROBOT, View


def robot_pixel_mask(panda, q, width, K, T_wc, H, W, dilate=6) -> np.ndarray:
    """Return an (H, W) bool mask of pixels covered by the robot's collision spheres at ``q``."""
    import cv2
    centers, radii = panda.spheres(q, width)[:2]
    T_cw = np.linalg.inv(np.asarray(T_wc, float))
    R, t = T_cw[:3, :3], T_cw[:3, 3]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    mask = np.zeros((H, W), np.uint8)
    pc = centers @ R.T + t                      # (S, 3) camera frame, z forward
    for (xc, yc, zc), r in zip(pc, radii):
        if zc <= 1e-3:
            continue
        u = int(round(fx * xc / zc + cx))
        v = int(round(fy * yc / zc + cy))
        rp = int(round(r * fx / zc)) + dilate
        if rp > 0:
            cv2.circle(mask, (u, v), rp, 1, -1)
    return mask.astype(bool)


def _backproject(ys, xs, depth, K, T_wc):
    """Return (N, 3) world points for pixels ``(ys, xs)`` with finite positive depth."""
    z = depth[ys, xs].astype(np.float64)
    ok = np.isfinite(z) & (z > 0)
    ys, xs, z = ys[ok], xs[ok], z[ok]
    if len(z) == 0:
        return np.zeros((0, 3))
    Kinv = np.linalg.inv(np.asarray(K, float))
    uv1 = np.stack([xs, ys, np.ones_like(xs)], 0).astype(np.float64)
    pc = (Kinv @ uv1) * z[None, :]                                     # (3, N) camera frame
    pw = np.asarray(T_wc, float)[:3, :3] @ pc + np.asarray(T_wc, float)[:3, 3:4]
    return pw.T


class SamSegmenter:
    """Turn SAM's automatic masks into an instance map filtered by table, workspace and robot."""

    def __init__(self, checkpoint, model_type="vit_b", device="cpu",
                 points_per_side=16, pred_iou_thresh=0.86, stability_score_thresh=0.9,
                 min_mask_region_area=300):
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=device)
        self.gen = SamAutomaticMaskGenerator(
            sam, points_per_side=points_per_side, pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh, min_mask_region_area=min_mask_region_area)

    def segment(self, rgb, depth, K, T_wc, robot_mask=None, table_z=0.006, max_z=0.30,
                roi=None, min_area=300, max_area_frac=0.35) -> np.ndarray:
        """Return an (H, W) int32 map: object id >= 0, -1 table or background, -2 robot."""
        rgb = np.asarray(rgb)[..., :3].astype(np.uint8)
        H, W = rgb.shape[:2]
        masks = self.gen.generate(rgb)
        seg = np.full((H, W), -1, np.int32)
        # paint large masks first so a small object on top keeps its own id
        masks = sorted(masks, key=lambda m: -int(m["area"]))
        next_id = 0
        for m in masks:
            area = int(m["area"])
            if area < min_area or area > max_area_frac * H * W:
                continue
            sm = m["segmentation"]
            ys, xs = np.nonzero(sm)
            pw = _backproject(ys, xs, depth, K, T_wc)
            if len(pw) < 20:
                continue
            zmed = float(np.median(pw[:, 2]))
            cxy = np.median(pw[:, :2], axis=0)
            if zmed < table_z or zmed > max_z:
                continue
            if roi is not None:
                lo, hi = roi
                if not (lo[0] <= cxy[0] <= hi[0] and lo[1] <= cxy[1] <= hi[1]):
                    continue
            seg[sm] = next_id
            next_id += 1
        if robot_mask is not None:
            seg[robot_mask] = SEG_ROBOT
        return seg

    def segment_view(self, view: View, rgb, panda=None, q=None, width=0.08, **kw) -> View:
        """Return a copy of ``view`` with ``seg`` replaced by the SAM map."""
        rmask = None
        if panda is not None and q is not None:
            rmask = robot_pixel_mask(panda, q, width, view.K, view.T_wc, *view.depth.shape[:2])
        seg = self.segment(rgb, view.depth, view.K, view.T_wc, robot_mask=rmask, **kw)
        return View(depth=view.depth, K=view.K, T_wc=view.T_wc, seg=seg, name=view.name, meta=dict(view.meta))


def instance_centroids(seg, depth, K, T_wc) -> dict:
    """Return the world centroid of each instance id (>= 0) in ``seg``."""
    out = {}
    for i in np.unique(seg[seg >= 0]):
        ys, xs = np.nonzero(seg == i)
        pw = _backproject(ys, xs, depth, K, T_wc)
        if len(pw):
            out[int(i)] = pw.mean(0)
    return out


def detect_basket(rgb, depth, K, T_wc, min_px=400):
    """Find the red basket in RGB-D; return world ``(centre_xy (2,), rim_z, n_px)`` or None."""
    rgb = np.asarray(rgb)[..., :3].astype(np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red = (r > 0.4) & (r - g > 0.18) & (r - b > 0.18)
    if int(red.sum()) < min_px:
        return None
    ys, xs = np.nonzero(red)
    pw = _backproject(ys, xs, depth, K, T_wc)
    if len(pw) < min_px // 2:
        return None
    # box centre, unlike the median, is not biased by which walls the camera sees most
    lo = np.percentile(pw[:, :2], 2, axis=0)
    hi = np.percentile(pw[:, :2], 98, axis=0)
    center = 0.5 * (lo + hi)
    rim_z = float(np.percentile(pw[:, 2], 90))
    return center, rim_z, int(red.sum())


def match_target(seg, depth, K, T_wc, target_xy) -> int | None:
    """Return the instance id whose centroid is nearest the world ``(x, y)`` target."""
    cents = instance_centroids(seg, depth, K, T_wc)
    if not cents:
        return None
    txy = np.asarray(target_xy, float)[:2]
    return min(cents, key=lambda i: np.linalg.norm(cents[i][:2] - txy))
