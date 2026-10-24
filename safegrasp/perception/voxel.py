"""3-state voxel map (FREE / OCCUPIED / UNKNOWN) built by projective carving of depth views, and
signed distance fields of voxel masks (the d(x), grad d(x) used by the CBF barriers)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter

from .pointcloud import pixel_rays, transform_points
from .types import SEG_BACKGROUND, SEG_ROBOT, View

__all__ = [
    "FREE", "OCC", "UNK",
    "VoxelGrid", "build_occupancy", "DistanceField",
    "unknown_mask_for_target", "obstacle_mask", "target_mask", "mask_points", "label_counts",
]

FREE, OCC, UNK = 0, 1, 2


@dataclass(eq=False)
class VoxelGrid:
    """Axis-aligned regular grid; voxel centre = origin + (idx + 0.5) res [m].
    Default covers x in [0.2, 0.9], y in [-0.4, 0.4], z in [0, 0.4] at 1 cm."""

    origin: np.ndarray = field(default_factory=lambda: np.array([0.20, -0.40, 0.0]))
    res: float = 0.01
    shape: tuple = (70, 80, 40)
    _centers: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    _centers_key: tuple | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        self.res = float(self.res)
        self.shape = tuple(int(s) for s in np.asarray(self.shape).reshape(3))

    def __eq__(self, other) -> bool:
        # the generated __eq__ would compare ndarrays (ambiguous truth value)
        if not isinstance(other, VoxelGrid):
            return NotImplemented
        return (self.shape == other.shape and self.res == other.res
                and bool(np.array_equal(self.origin, other.origin)))

    __hash__ = None  # mutable

    @classmethod
    def from_bounds(cls, lo, hi, res=0.01) -> "VoxelGrid":
        """Smallest grid with voxel size ``res`` starting at ``lo`` that covers ``[lo, hi]``."""
        lo = np.asarray(lo, dtype=np.float64).reshape(3)
        hi = np.asarray(hi, dtype=np.float64).reshape(3)
        shape = np.maximum(np.ceil((hi - lo) / res - 1e-9), 1).astype(int)
        return cls(origin=lo, res=res, shape=tuple(shape))

    @property
    def shape_arr(self) -> np.ndarray:
        return np.asarray(self.shape, dtype=np.int64)

    @property
    def lo(self) -> np.ndarray:
        """Lower corner of the grid box (= origin)."""
        return self.origin.copy()

    @property
    def hi(self) -> np.ndarray:
        """Upper corner of the grid box."""
        return self.origin + self.res * self.shape_arr

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    @property
    def voxel_volume(self) -> float:
        return self.res ** 3

    def world_to_index(self, pts) -> np.ndarray:
        """(N,3) world points -> (N,3) int64 voxel indices; may be out of bounds."""
        pts = np.asarray(pts, dtype=np.float64)
        return np.floor((pts - self.origin) / self.res).astype(np.int64)

    def index_to_world(self, idx) -> np.ndarray:
        """(N,3) voxel indices -> (N,3) world coordinates of voxel centres."""
        return self.origin + (np.asarray(idx, dtype=np.float64) + 0.5) * self.res

    def in_bounds(self, idx) -> np.ndarray:
        idx = np.asarray(idx)
        return np.all((idx >= 0) & (idx < self.shape_arr), axis=-1)

    def contains(self, pts) -> np.ndarray:
        """(N,) bool: point lies inside the grid box."""
        pts = np.asarray(pts, dtype=np.float64)
        return np.all((pts >= self.lo) & (pts < self.hi), axis=-1)

    def ravel_index(self, idx) -> np.ndarray:
        """(N,3) in-bounds indices -> (N,) flat C-order indices."""
        idx = np.asarray(idx, dtype=np.int64)
        nx, ny, nz = self.shape
        return (idx[..., 0] * ny + idx[..., 1]) * nz + idx[..., 2]

    def centers(self, flat: bool = False) -> np.ndarray:
        """Voxel centres (nx,ny,nz,3), or (N,3) C-order if ``flat``; cached, read-only."""
        key = (np.asarray(self.origin, dtype=np.float64).tobytes(), float(self.res),
               tuple(self.shape))
        if self._centers is None or self._centers_key != key:   # (re)build if grid was edited
            self._centers_key = key
            axes = [self.origin[a] + (np.arange(self.shape[a]) + 0.5) * self.res for a in range(3)]
            c = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
            c.flags.writeable = False
            self._centers = c
        return self._centers.reshape(-1, 3) if flat else self._centers


def _majority_vote(lin: np.ndarray, ids: np.ndarray, min_points: int = 1):
    """Majority label per voxel: lin (N,) flat voxel indices, ids (N,) labels >= -1.
    Returns (voxels (M,), labels (M,)); ties go to the larger id, so an object beats background."""
    if lin.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int32)
    s = ids.astype(np.int64) + 1                       # -1 -> 0, object k -> k+1
    nid = int(s.max()) + 1
    key = lin * nid + s
    uk, cnt = np.unique(key, return_counts=True)       # sorted by (voxel, label)
    vox, lab = uk // nid, uk % nid - 1
    # after sorting, each voxel's last entry is its max-count label
    order = np.lexsort((cnt, vox))                     # by voxel, then count (stable)
    vox, lab, cnt = vox[order], lab[order], cnt[order]
    last = np.r_[vox[1:] != vox[:-1], True]
    if min_points > 1:
        tot = np.add.reduceat(cnt, np.flatnonzero(np.r_[True, vox[1:] != vox[:-1]]))
        keep = tot >= min_points
        return vox[last][keep], lab[last][keep].astype(np.int32)
    return vox[last], lab[last].astype(np.int32)


def build_occupancy(views: list[View], grid: VoxelGrid, trunc=0.01, min_z=0.004, *,
                    bg_min_z=0.015, table_trunc=0.003, point_stride=1, min_points=1,
                    carve_robot=False, inf_is_free=False, robot_spheres=None, robot_pad=0.01,
                    fill_radius=1, table_analytic=True):
    """Projective carving of a 3-state voxel map; returns labels (nx,ny,nz) int8 and inst int32.

    FREE: some view sees a valid surface behind the voxel (z_cam < D - trunc). OCC: holds object
    points or background points above bg_min_z (inst = majority seg id, else -1). Rest is UNK.
    Lengths in metres; bg_min_z > min_z so table noise does not make speckle obstacles.
    robot_spheres: (centres (S,3), radii (S,), ...) at capture time; voxels inside become FREE.
    """
    C = grid.centers(flat=True)                          # (M,3)
    M = C.shape[0]
    below = C[:, 2] < min_z                              # analytic table => FREE
    free = below.copy()
    pending = np.flatnonzero(~below)                     # voxels not yet carved
    occ_pts, occ_ids = [], []
    if table_trunc is None:
        table_trunc = trunc

    for view in views:
        H, W = view.depth.shape
        seg = view.seg
        R, t = view.T_wc[:3, :3], view.T_wc[:3, 3]
        valid = np.isfinite(view.depth) & (view.depth > 0)
        robot = seg == SEG_ROBOT
        depth = np.where(valid, view.depth, 0.0).astype(np.float64)

        # world height of each pixel's surface: z_w = t_z + D * d_z (rays have unit camera z)
        rays = pixel_rays(view.K, H, W)                   # (H,W,3)
        dz = np.einsum("hwi,i->hw", rays, R[2])
        zw = t[2] + depth * dz
        bg = seg == SEG_BACKGROUND
        table = valid & bg & (zw < bg_min_z)

        # a voxel is FREE iff z_cam < thr(u, v)
        carve_px = valid if carve_robot else (valid & ~robot)
        thr = np.where(carve_px, depth - trunc, -np.inf)
        if table_analytic:
            # table pixels: the exact z = 0 plane depth -t_z / d_z replaces the noisy measurement
            ok = table & (dz < -1e-9)
            thr[ok] = -t[2] / dz[ok] - table_trunc
            thr[table & ~ok] = depth[table & ~ok] - table_trunc
        else:
            thr[table] = depth[table] - table_trunc
        if fill_radius > 0:
            # give dropped pixels the nearby min threshold so they leave no UNK speckles
            hole = ~valid
            src = np.where(carve_px, thr, np.inf)
            if not carve_robot:
                src[robot] = -np.inf
            filled = minimum_filter(src, size=2 * fill_radius + 1, mode="nearest")[hole]
            thr[hole] = np.where(np.isfinite(filled), filled, -np.inf)
        if inf_is_free:
            thr[np.isposinf(view.depth)] = np.inf

        # project not-yet-free voxel centres into the camera, p = R^T (x - t)
        if pending.size:
            pc = np.einsum("ni,ij->nj", C[pending] - t, R)
            z = pc[:, 2]
            front = z > 1e-6
            zf = np.where(front, z, 1.0)
            uvw = np.einsum("ij,nj->ni", view.K, pc)
            u = np.floor(uvw[:, 0] / zf + 0.5).astype(np.int64)
            v = np.floor(uvw[:, 1] / zf + 0.5).astype(np.int64)
            ii = np.flatnonzero(front & (u >= 0) & (u < W) & (v >= 0) & (v < H))
            carved = ii[z[ii] < thr.ravel()[v[ii] * W + u[ii]]]
            free[pending[carved]] = True
            keep = np.ones(pending.size, dtype=bool)
            keep[carved] = False
            pending = pending[keep]

        occ_px = valid & ~robot & ((seg >= 0) | (bg & (zw > bg_min_z)))
        if point_stride > 1:
            sub = np.zeros_like(occ_px)
            sub[::point_stride, ::point_stride] = True
            occ_px &= sub
        rr, cc = np.nonzero(occ_px)
        occ_pts.append(transform_points(view.T_wc, rays[rr, cc] * depth[rr, cc, None]))
        occ_ids.append(seg[rr, cc])

    if robot_spheres is not None:
        # extra items (e.g. jac, link ids from Panda.spheres()) are ignored
        free |= _in_spheres_mask(grid, robot_spheres[0], robot_spheres[1], pad=robot_pad).ravel()

    labels = np.full(M, UNK, dtype=np.int8)
    inst = np.full(M, -1, dtype=np.int32)
    labels[free] = FREE

    if occ_pts:
        P = np.concatenate(occ_pts)
        I = np.concatenate(occ_ids)
        idx = grid.world_to_index(P)
        ok = grid.in_bounds(idx)
        vox, lab = _majority_vote(grid.ravel_index(idx[ok]), I[ok], min_points)
        labels[vox] = OCC
        inst[vox] = lab

    labels[below] = FREE                                 # analytic table overrides
    inst[below] = -1
    return labels.reshape(grid.shape), inst.reshape(grid.shape)


def _in_spheres_mask(grid: VoxelGrid, centers, radii, pad=0.0) -> np.ndarray:
    """(nx,ny,nz) bool: voxel centre within r_s + pad of any sphere centre."""
    out = np.zeros(grid.shape, dtype=bool)
    C = grid.centers()
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (len(centers),))
    for c, r in zip(centers, radii + pad):
        lo = np.maximum(grid.world_to_index(c - r), 0)
        hi = np.minimum(grid.world_to_index(c + r) + 1, grid.shape_arr)
        if np.any(hi <= lo):
            continue
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        out[sl] |= np.sum((C[sl] - c) ** 2, axis=-1) <= r * r
    return out


def label_counts(labels) -> dict:
    """Number of FREE / OCC / UNK voxels."""
    b = np.bincount(np.asarray(labels).ravel().astype(np.int64), minlength=3)
    return {"free": int(b[FREE]), "occ": int(b[OCC]), "unk": int(b[UNK])}


def obstacle_mask(labels, inst, target_id) -> np.ndarray:
    """OCC voxels not belonging to the target (includes unexpected background surfaces)."""
    return (labels == OCC) & (inst != target_id)


def target_mask(labels, inst, target_id) -> np.ndarray:
    """OCC voxels of the target object."""
    return (labels == OCC) & (inst == target_id)


def mask_points(mask, grid: VoxelGrid) -> np.ndarray:
    """(M,3) centres of the voxels in a boolean mask."""
    return grid.centers()[np.asarray(mask, dtype=bool)]


def _points_in_box(pts, T_world_box, half_extents) -> np.ndarray:
    """(N,) bool: points inside the oriented box ``|R^T (p - t)| <= half_extents``."""
    T = np.asarray(T_world_box, dtype=np.float64)
    he = np.broadcast_to(np.asarray(half_extents, dtype=np.float64), (3,))
    local = np.einsum("nj,ji->ni", pts - T[:3, 3], T[:3, :3])    # R^T (p - t)
    return np.all(np.abs(local) <= he, axis=1)


def _in_target_shadow(pts, views, target_id, dilate=2) -> np.ndarray:
    """(N,) bool: point projects onto a target pixel (dilated by ``dilate`` px) in some view.
    For an UNK voxel this means it lies behind the target, i.e. in its shadow."""
    out = np.zeros(len(pts), dtype=bool)
    for view in views:
        tgt_px = view.seg == target_id
        if not tgt_px.any():
            continue
        if dilate > 0:
            tgt_px = maximum_filter(tgt_px, size=2 * int(dilate) + 1, mode="constant")
        H, W = tgt_px.shape
        R, t = view.T_wc[:3, :3], view.T_wc[:3, 3]
        pc = np.einsum("ni,ij->nj", pts - t, R)               # camera frame, R^T (x - t)
        z = pc[:, 2]
        front = z > 1e-6
        zf = np.where(front, z, 1.0)
        uvw = np.einsum("ij,nj->ni", view.K, pc)
        u = np.floor(uvw[:, 0] / zf + 0.5).astype(np.int64)
        v = np.floor(uvw[:, 1] / zf + 0.5).astype(np.int64)
        ii = np.flatnonzero(front & (u >= 0) & (u < W) & (v >= 0) & (v < H))
        out[ii[tgt_px[v[ii], u[ii]]]] = True
    return out


def unknown_mask_for_target(labels, inst, target_id, grid: VoxelGrid, carve_radius=0.03,
                            extra_boxes=(), roi=None, views=None, shadow_dilate=2) -> np.ndarray:
    """UNK voxels the robot must avoid while grasping target_id.

    The fingers must enter the target's own shadow, so UNK voxels within carve_radius [m] of the
    target and in its shadow (all of them if views is None) are removed, as are voxels inside
    extra_boxes, each (T_world_box (4,4), half_extents (3,)). roi: optional (lo, hi) box to keep.
    """
    unk = np.asarray(labels) == UNK
    if roi is not None:
        lo, hi = (np.asarray(a, dtype=np.float64) for a in roi)
        C = grid.centers()
        unk &= np.all((C >= lo) & (C <= hi), axis=-1)
    tgt = target_mask(labels, inst, target_id)

    if carve_radius > 0 and tgt.any():
        # EDT only over the target's bounding box grown by the carve radius
        pad = int(np.ceil(carve_radius / grid.res)) + 1
        nz_idx = np.argwhere(tgt)
        lo = np.maximum(nz_idx.min(0) - pad, 0)
        hi = np.minimum(nz_idx.max(0) + pad + 1, grid.shape_arr)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        dist = distance_transform_edt(~tgt[sl], sampling=grid.res)
        near = unk[sl] & (dist <= carve_radius + 1e-9)
        if views is not None and near.any():
            idx = np.argwhere(near) + lo
            shadow = _in_target_shadow(grid.index_to_world(idx), views, target_id, shadow_dilate)
            near[tuple((idx[~shadow] - lo).T)] = False
        unk[sl] &= ~near

    for box in extra_boxes:
        T_box, he = box
        flat = np.flatnonzero(unk)
        if flat.size == 0:
            break
        inside = _points_in_box(grid.centers(flat=True)[flat], T_box, he)
        unk.flat[flat[inside]] = False
    return unk


class DistanceField:
    """Signed Euclidean distance to a voxel mask [m], negative inside, with trilinear interpolation.

    The zero level sits on the mask's voxel faces; error is up to ~0.6 res, so use margins >= 1 res.
    Outside the grid box, d = sqrt(e^2 + max(d(c), 0)^2) with c the nearest box point, e = |p - c|:
    a lower bound that grows away from the box. Non-finite points give d = nan, zero gradient.
    """

    def __init__(self, dist: np.ndarray, grid: VoxelGrid, max_dist: float = 0.3,
                 empty: bool = False):
        self.grid = grid
        self.max_dist = float(max_dist)
        self.empty = bool(empty)
        self.dist = np.asarray(dist, dtype=np.float64)       # (nx,ny,nz)
        if self.empty:
            self.grad = np.zeros(self.dist.shape + (3,))
        else:
            # an axis with a single voxel has zero derivative
            g = [np.gradient(self.dist, grid.res, axis=a) if self.dist.shape[a] > 1
                 else np.zeros_like(self.dist) for a in range(3)]
            self.grad = np.stack(g, axis=-1)                 # (nx,ny,nz,3)
        # packed [d, gx, gy, gz] table for a single gather per corner
        self._table = np.concatenate([self.dist[..., None], self.grad], axis=-1).reshape(-1, 4)
        nx, ny, nz = grid.shape
        # flat offsets of the 8 cell corners (0 along degenerate axes)
        sx, sy, sz = (ny * nz if nx > 1 else 0), (nz if ny > 1 else 0), (1 if nz > 1 else 0)
        c = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        self._corner_bits = c.astype(np.float64)              # (8,3)
        self._corner_off = c @ np.array([sx, sy, sz])        # (8,)
        self._imax = np.maximum(grid.shape_arr - 2, 0)

    @classmethod
    def from_mask(cls, mask, grid: VoxelGrid, max_dist=0.3, signed=True) -> "DistanceField":
        """Signed distance field of a boolean (nx,ny,nz) mask (``signed=False``: 0 inside)."""
        mask = np.asarray(mask, dtype=bool).reshape(grid.shape)
        if not mask.any():
            return cls(np.full(grid.shape, float(max_dist)), grid, max_dist, empty=True)
        h = 0.5 * grid.res
        d_out = distance_transform_edt(~mask, sampling=grid.res)
        if signed and mask.all():
            d = np.full(grid.shape, -float(max_dist))
        elif signed:
            d_in = distance_transform_edt(mask, sampling=grid.res)
            d = np.where(mask, -(d_in - h), d_out - h)
        else:
            d = np.where(mask, 0.0, d_out - h)
        np.clip(d, -max_dist, max_dist, out=d)
        return cls(d, grid, max_dist)

    @classmethod
    def empty_field(cls, grid: VoxelGrid, max_dist=0.3) -> "DistanceField":
        return cls(np.full(grid.shape, float(max_dist)), grid, max_dist, empty=True)

    def _interp(self, p: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of [d, grad] at points inside the grid box -> (N,4)."""
        g = self.grid
        f = (p - g.origin) / g.res - 0.5                     # continuous centre coordinates
        i0 = np.clip(np.floor(f).astype(np.int64), 0, self._imax)
        t = np.clip(f - i0, 0.0, 1.0)                        # (N,3) cell-local coordinates
        base = g.ravel_index(i0)
        vals = self._table[base[:, None] + self._corner_off[None, :]]      # (N,8,4)
        b = self._corner_bits                                # weights prod_a (b t + (1-b)(1-t))
        w = np.prod(b[None] * t[:, None, :] + (1 - b[None]) * (1 - t[:, None, :]), axis=2)
        return np.einsum("nc,ncd->nd", w, vals)

    def query(self, pts):
        """Distance d (...,) [m] and unit gradient (...,3) at world points pts (...,3).
        d is negative inside the mask and <= max_dist; grad is zero where undefined."""
        pts = np.asarray(pts, dtype=np.float64)
        shp = pts.shape[:-1]
        p = pts.reshape(-1, 3)
        n = p.shape[0]
        if self.empty or n == 0:
            return np.full(shp, self.max_dist), np.zeros(shp + (3,))
        bad = ~np.all(np.isfinite(p), axis=1)
        if bad.any():                                        # NaN / inf query points
            p = np.where(bad[:, None], 0.0, p)

        g = self.grid
        c = np.clip(p, g.lo, g.hi)                           # closest point of the grid box
        out = self._interp(c)
        d, grad = out[:, 0], out[:, 1:]

        diff = p - c
        e = np.sqrt(np.einsum("ni,ni->n", diff, diff))       # distance to the grid box
        outside = e > 0
        if outside.any():
            eo = e[outside]
            nrm_out = diff[outside] / eo[:, None]            # outward box normal n
            dc = np.maximum(d[outside], 0.0)                 # d(c), mask lies inside the box
            gc = grad[outside]
            gn = np.linalg.norm(gc, axis=1)
            gc = np.where(gn[:, None] > 1e-6, gc / np.maximum(gn, 1e-12)[:, None], 0.0)
            d[outside] = np.sqrt(eo * eo + dc * dc)          # Pythagorean lower bound
            grad[outside] = eo[:, None] * nrm_out + dc[:, None] * gc

        nrm = np.linalg.norm(grad, axis=1)
        ok = nrm > 1e-6
        grad[ok] /= nrm[ok, None]
        grad[~ok] = 0.0
        d = np.minimum(d, self.max_dist)
        if bad.any():
            d[bad] = np.nan
            grad[bad] = 0.0
        return d.reshape(shp), grad.reshape(shp + (3,))

    def __call__(self, pts):
        return self.query(pts)
