"""Propose 6-DoF grasps near a target with VGN on a TSDF (voxel distance grid) from depth.

Repo and weights default to data/vgn; set VGN_SRC and VGN_MODEL to change them. The helpers
copy predict/process/select from vgn.detection to avoid its ROS import.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import ndimage

from .types import Grasp, canonicalize_about_approach

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
VGN_SRC = os.environ.get("VGN_SRC", os.path.join(_DATA, "vgn", "src"))
VGN_MODEL = os.environ.get("VGN_MODEL", os.path.join(_DATA, "vgn", "data", "models", "vgn_conv.pth"))


class VGNProposer:
    def __init__(self, model=VGN_MODEL, src=VGN_SRC, device="cpu", size=0.30, res=40,
                 threshold=0.90, near_target=0.04):
        if src not in sys.path:
            sys.path.insert(0, src)
        import torch
        from pathlib import Path
        from vgn.networks import load_network
        from vgn.perception import CameraIntrinsic, TSDFVolume
        from vgn.utils.transform import Rotation, Transform
        self.torch, self.Rotation, self.Transform = torch, Rotation, Transform
        self.CameraIntrinsic, self.TSDFVolume = CameraIntrinsic, TSDFVolume
        self.net = load_network(Path(model), device)
        self.net.eval()
        self.device, self.size, self.res = device, size, res
        self.threshold, self.near_target = threshold, near_target

    def _predict(self, grid):
        x = self.torch.from_numpy(grid).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            q, r, w = self.net(x)
        return q.cpu().squeeze().numpy(), r.cpu().squeeze().numpy(), w.cpu().squeeze().numpy()

    @staticmethod
    def _process(grid, qual, rot, width, sigma=1.0, min_w=1.33, max_w=9.33):
        tsdf = grid.squeeze()
        qual = ndimage.gaussian_filter(qual, sigma=sigma, mode="nearest")
        outside = tsdf > 0.5
        inside = (1e-3 < tsdf) & (tsdf < 0.5)
        valid = ndimage.binary_dilation(outside, iterations=2, mask=~inside)
        qual[~valid] = 0.0
        qual[(width < min_w) | (width > max_w)] = 0.0
        return qual

    def _select(self, qual, rot, width, maxf=4):
        qual = qual.copy()
        qual[qual < self.threshold] = 0.0
        qual = np.where(qual == ndimage.maximum_filter(qual, size=maxf), qual, 0.0)
        out = []
        for i, j, k in np.argwhere(qual > 0):
            out.append((float(qual[i, j, k]), self.Rotation.from_quat(rot[:, i, j, k]),
                        np.array([i, j, k], float), float(width[i, j, k])))
        return out

    def propose(self, depth, K, T_wc, target_pts, cfg=None):
        target_pts = np.asarray(target_pts, float).reshape(-1, 3)
        if len(target_pts) < 20:
            return []
        c = target_pts[:, :2].mean(0)
        voxel = self.size / self.res
        O = np.array([c[0] - self.size / 2, c[1] - self.size / 2, 0.0])

        H, W = depth.shape
        intr = self.CameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        tsdf = self.TSDFVolume(self.size, self.res)
        T_world_vol = np.eye(4); T_world_vol[:3, 3] = O
        extr = np.linalg.inv(np.asarray(T_wc, float)) @ T_world_vol        # volume -> camera
        tsdf.integrate(depth.astype(np.float32), intr, self.Transform.from_matrix(extr))
        grid = tsdf.get_grid()

        qual, rot, width = self._predict(grid)
        qual = self._process(grid, qual, rot, width)
        cand = self._select(qual, rot, width)

        grasps = []
        for score, ori, pvox, wvox in sorted(cand, key=lambda x: -x[0]):
            pos = O + pvox * voxel
            if np.min(np.linalg.norm(target_pts - pos, axis=1)) > self.near_target:
                continue
            R = ori.as_matrix()                                  # col2 = approach (VGN +z)
            # the arm comes from above, so skip upward approaches
            if R[2, 2] > -0.1:
                continue
            width_m = float(np.clip(wvox * voxel, 0.02, 0.08))
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos
            T = canonicalize_about_approach(T, (0.0, 0.0))
            grasps.append(Grasp(T=T, width=width_m, score=score, method="vgn",
                                info=dict(quality=score, depth=0.0, grip_width=width_m)))
        return grasps
