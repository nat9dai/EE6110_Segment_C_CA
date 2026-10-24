"""Top-down PCA grasp baseline (td_nom): close straight down across the short side of the target's footprint.

No collision filtering; with obstacle_pts the grasps only get info['collides'].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gripper import collides_batch, get_geometry
from .types import Grasp, canonicalize_about_approach, make_cfg, make_pose


@dataclass
class TopDownConfig:
    width_pad: float = 0.02            # width = minor extent + pad
    min_width: float = 0.02
    max_width: float = 0.08
    max_graspable: float = 0.075       # minor extent above which the grasp is hopeless
    max_insert: float = 0.03           # TCP at most this far below the top
    insert_frac: float = 0.5           # ... or this fraction of the height above the table
    min_tcp_z: float = 0.012           # keep fingertips off the table (tip is 9 mm below TCP)
    table_z: float = 0.0
    pct: float = 1.0                   # robust extents use [pct, 100-pct] percentiles
    center_mode: str = "midrange"      # 'midrange' (PCA box centre, less biased by partial views) or 'mean'
    yaw_variants_deg: tuple = (10.0, -10.0)
    depth_variants: tuple = (0.01,)    # shallower variants raised by these amounts (m)
    base_xy: tuple = (0.0, 0.0)
    method: str = "topdown"


def _pca2d(xy: np.ndarray):
    """Return (mean (2,), major (2,), minor (2,), eigvals (2,) descending)."""
    mu = xy.mean(0)
    C = np.cov((xy - mu).T) if len(xy) > 2 else np.eye(2) * 1e-6
    evals, evecs = np.linalg.eigh(C)            # ascending
    major, minor = evecs[:, 1], evecs[:, 0]
    # deterministic sign: major has +x (or +y) component
    if major[0] < 0 or (abs(major[0]) < 1e-9 and major[1] < 0):
        major = -major
    minor = np.array([-major[1], major[0]])     # right-handed (major, minor)
    return mu, major, minor, evals[::-1]


def _extent(xy: np.ndarray, axis: np.ndarray, pct: float):
    s = xy @ axis
    lo, hi = np.percentile(s, [pct, 100.0 - pct]) if pct > 0 else (s.min(), s.max())
    return float(hi - lo), float(0.5 * (lo + hi))


def _topdown_pose(center_xy, closing_xy, z) -> np.ndarray:
    """TCP pose with hand z = -world z and hand y = closing (horizontal)."""
    y = np.array([closing_xy[0], closing_xy[1], 0.0])
    y /= np.linalg.norm(y)
    zax = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, zax)
    return make_pose(np.stack([x, y, zax], axis=1), np.array([center_xy[0], center_xy[1], z]))


def propose_topdown(target_pts, rng=None, cfg=None, obstacle_pts=None) -> list[Grasp]:
    """Top-down PCA grasps on the target cloud (N,3), best first; [] if under 3 finite points.

    rng is unused (deterministic) and kept for a uniform interface.
    """
    cfg = make_cfg(TopDownConfig, cfg)
    P = np.asarray(target_pts, dtype=np.float64).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) < 3:
        return []
    xy = P[:, :2]
    mu, major, minor, evals = _pca2d(xy)
    e_major, c_major = _extent(xy, major, cfg.pct)
    e_minor, c_minor = _extent(xy, minor, cfg.pct)
    center = mu if cfg.center_mode == "mean" else c_major * major + c_minor * minor

    z_top = float(np.percentile(P[:, 2], 100.0 - cfg.pct)) if cfg.pct > 0 else float(P[:, 2].max())
    height = max(z_top - cfg.table_z, 0.0)
    z_tcp = max(z_top - min(cfg.max_insert, cfg.insert_frac * height), cfg.table_z + cfg.min_tcp_z)

    graspable = e_minor <= cfg.max_graspable
    # score in (0.5, 1] when graspable: thinner footprints score higher
    base = (1.0 - 0.5 * e_minor / cfg.max_graspable) if graspable else 0.02

    grasps: list[Grasp] = []

    def add(yaw_off, dz, penalty):
        c, s = np.cos(yaw_off), np.sin(yaw_off)
        ax = np.array([c * minor[0] - s * minor[1], s * minor[0] + c * minor[1]])
        ext, _ = _extent(xy, ax, cfg.pct)
        w = float(np.clip(ext + cfg.width_pad, cfg.min_width, cfg.max_width))
        z = max(z_tcp + dz, cfg.table_z + cfg.min_tcp_z)
        T = canonicalize_about_approach(_topdown_pose(center, ax, z), cfg.base_xy)
        info = dict(minor_extent=e_minor, major_extent=e_major, extent_along_closing=ext,
                    height=height, z_top=z_top, yaw_offset=float(yaw_off), depth_offset=float(dz),
                    graspable=float(graspable), elongation=float(e_major / max(e_minor, 1e-6)))
        grasps.append(Grasp(T=T, width=w, score=base - penalty, method=cfg.method, info=info))

    add(0.0, 0.0, 0.0)
    for k, yd in enumerate(cfg.yaw_variants_deg):
        add(np.deg2rad(yd), 0.0, 0.02 * (1 + k // 2))
    for k, dz in enumerate(cfg.depth_variants):
        add(0.0, float(dz), 0.03 * (k + 1))

    if obstacle_pts is not None and len(obstacle_pts):
        Ts = np.stack([g.T for g in grasps])
        ws = np.array([g.width for g in grasps])
        hit = collides_batch(Ts, ws, obstacle_pts, margin=0.003, geom=get_geometry())
        for g, h in zip(grasps, hit):
            g.info["collides"] = float(h)
    grasps.sort(key=lambda g: -g.score)
    return grasps
