"""Occlusion shadows: voxels a camera looks at but cannot see because a surface lies in front.
Unlike out-of-frustum UNKNOWN space, these may hide objects, so the occlusion-aware barrier uses
UNKNOWN ∩ shadow."""
from __future__ import annotations

import numpy as np

from safegrasp.perception.types import View
from safegrasp.perception.voxel import VoxelGrid


def shadow_mask(views: list[View], grid: VoxelGrid, trunc: float = 0.01) -> np.ndarray:
    C = grid.centers(flat=True).reshape(-1, 3)
    shadow = np.zeros(len(C), dtype=bool)
    for v in views:
        T_cw = np.linalg.inv(v.T_wc)
        pc = np.einsum("ij,nj->ni", T_cw[:3, :3], C) + T_cw[:3, 3]
        z = pc[:, 2]
        H, W = v.depth.shape
        ok = z > 1e-3
        u = np.full(len(C), -1, dtype=np.int64)
        w = np.full(len(C), -1, dtype=np.int64)
        u[ok] = np.round(v.K[0, 0] * pc[ok, 0] / z[ok] + v.K[0, 2]).astype(np.int64)
        w[ok] = np.round(v.K[1, 1] * pc[ok, 1] / z[ok] + v.K[1, 2]).astype(np.int64)
        ok &= (u >= 0) & (u < W) & (w >= 0) & (w < H)
        idx = np.flatnonzero(ok)
        D = v.depth[w[idx], u[idx]]
        seg = v.seg[w[idx], u[idx]]
        valid = np.isfinite(D) & (D > 0) & (seg != -2)
        behind = valid & (z[idx] > D + trunc)
        shadow[idx[behind]] = True
    return shadow.reshape(tuple(grid.shape))
