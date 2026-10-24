"""Propose top-down grasps on a target with GG-CNN2 run on a heightmap of the point cloud.

Repo and weights default to data/ggcnn; set GGCNN_REPO and GGCNN_WEIGHTS to change them.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .topdown import _topdown_pose
from .types import Grasp, canonicalize_about_approach

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
GGCNN_REPO = os.environ.get("GGCNN_REPO", os.path.join(_DATA, "ggcnn"))
GGCNN_WEIGHTS = os.environ.get(
    "GGCNN_WEIGHTS", os.path.join(GGCNN_REPO, "weights", "ggcnn2_weights_cornell", "epoch_50_cornell_statedict.pt"))


@dataclass
class GGCNNConfig:
    res: int = 300
    span: float = 0.30             # window side, m
    width_scale: float = 150.0
    smooth: float = 2.0
    max_grasps: int = 12
    nms_px: int = 14
    min_quality: float = 0.2
    insert: float = 0.02           # TCP depth below the object top, m
    min_tcp_z: float = 0.012
    min_width: float = 0.02
    max_width: float = 0.08
    width_pad: float = 0.015
    base_xy: tuple = (0.0, 0.0)
    method: str = "ggcnn"


class GGCNNProposer:
    """Load GG-CNN2 once and propose top-down grasps on a target, best first."""

    def __init__(self, weights=GGCNN_WEIGHTS, repo=GGCNN_REPO, device="cpu"):
        import sys
        if repo not in sys.path:
            sys.path.insert(0, repo)
        import torch
        from models.ggcnn2 import GGCNN2
        self.torch = torch
        self.net = GGCNN2()
        self.net.load_state_dict(torch.load(weights, map_location=device))
        self.net.eval()
        self.device = device

    def _heightmap(self, pts, lo, cell, res):
        """Top-down max-z heightmap; empty cells stay 0 (table)."""
        hm = np.zeros((res, res), np.float32)
        ij = np.floor((pts[:, :2] - lo) / cell).astype(np.int64)
        ok = (ij >= 0).all(1) & (ij < res).all(1)
        ij, z = ij[ok], pts[ok, 2].astype(np.float32)
        order = np.argsort(z)                      # write higher z last so each cell keeps its max
        flat = ij[order, 0] * res + ij[order, 1]
        hm.reshape(-1)[flat] = z[order]
        return hm

    def propose(self, target_pts, obstacle_pts=None, cfg: GGCNNConfig | None = None):
        from scipy.ndimage import gaussian_filter
        cfg = cfg or GGCNNConfig()
        P = np.asarray(target_pts, float).reshape(-1, 3)
        P = P[np.all(np.isfinite(P), axis=1)]
        if len(P) < 20:
            return []
        c = P[:, :2].mean(0)
        lo = c - cfg.span / 2.0
        cell = cfg.span / cfg.res
        allp = P if obstacle_pts is None or not len(obstacle_pts) else np.concatenate([P, np.asarray(obstacle_pts, float)])
        hm = self._heightmap(allp, lo, cell, cfg.res)
        tgt_hm = self._heightmap(P, lo, cell, cfg.res)
        tmask = tgt_hm > 1e-4

        # GG-CNN expects depth from above, so higher points must be nearer
        depth = -hm.astype(np.float32)
        depth = np.clip(depth - depth.mean(), -1.0, 1.0)
        x = self.torch.from_numpy(depth[None, None]).float().to(self.device)
        with self.torch.no_grad():
            pos, cosm, sinm, widm = self.net(x)
        q = gaussian_filter(pos[0, 0].cpu().numpy(), cfg.smooth)
        ang = 0.5 * np.arctan2(sinm[0, 0].cpu().numpy(), cosm[0, 0].cpu().numpy())
        wpx = widm[0, 0].cpu().numpy() * cfg.width_scale
        q = q * tmask

        grasps = []
        qf = q.copy()
        for _ in range(cfg.max_grasps):
            idx = int(np.argmax(qf))
            quality = float(qf.reshape(-1)[idx])
            if quality < cfg.min_quality:
                break
            i, j = divmod(idx, cfg.res)                # i indexes world x, j world y
            # clear the area so the next peak is a different grasp
            i0, i1 = max(0, i - cfg.nms_px), min(cfg.res, i + cfg.nms_px)
            j0, j1 = max(0, j - cfg.nms_px), min(cfg.res, j + cfg.nms_px)
            qf[i0:i1, j0:j1] = 0.0

            wx = lo[0] + (i + 0.5) * cell
            wy = lo[1] + (j + 0.5) * cell
            wz = float(hm[i, j])
            theta = float(ang[i, j])
            width = float(np.clip(wpx[i, j] * cell + cfg.width_pad, cfg.min_width, cfg.max_width))
            z_tcp = max(wz - cfg.insert, cfg.min_tcp_z)
            closing = np.array([np.cos(theta), np.sin(theta)])
            T = canonicalize_about_approach(_topdown_pose((wx, wy), closing, z_tcp), cfg.base_xy)
            grasps.append(Grasp(T=T, width=width, score=quality, method=cfg.method,
                                info=dict(quality=quality, theta=theta, z_top=wz, depth=cfg.insert,
                                          grip_width=width)))
        return grasps
