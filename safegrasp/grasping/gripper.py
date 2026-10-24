"""Box model of the Franka Panda gripper in the TCP frame, and point-cloud collision tests.

Defaults come from the Isaac Lab collision meshes; kinematics/panda_gripper_geometry.json overrides them.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np

from .types import MAX_WIDTH, TCP_OFFSET

GEOMETRY_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "kinematics", "panda_gripper_geometry.json")


@dataclass
class GripperGeometry:
    """Box model of the Panda hand in the TCP frame (metres)."""
    # hand body; measured mesh: x +-0.0316, y [-0.104, 0.100], hand z [-0.026, 0.066]
    palm_lo: np.ndarray = field(default_factory=lambda: np.array([-0.032, -0.105, -TCP_OFFSET - 0.026]))
    palm_hi: np.ndarray = field(default_factory=lambda: np.array([0.032, 0.105, -0.037]))
    # lower panda_link7, symmetric in y so the Rz(pi) flip is harmless
    wrist_lo: np.ndarray | None = field(default_factory=lambda: np.array([-0.045, -0.09, -0.160]))
    wrist_hi: np.ndarray | None = field(default_factory=lambda: np.array([0.045, 0.09, -TCP_OFFSET - 0.026]))
    # +y finger; the -y one is mirrored
    finger_x: np.ndarray = field(default_factory=lambda: np.array([-0.011, 0.011]))
    finger_z: np.ndarray = field(default_factory=lambda: np.array([-0.045, 0.009]))
    finger_thickness: float = 0.027      # outer face at y = w/2 + offset + thickness
    finger_offset: float = 0.0           # inner face at y = w/2 + offset
    pad_z: np.ndarray = field(default_factory=lambda: np.array([-0.020, 0.006]))
    closing_x_scale: float = 1.5         # closing region x-extent = finger_x * scale
    max_width: float = MAX_WIDTH
    tcp_offset: float = TCP_OFFSET
    source: str = "defaults"

    def __post_init__(self):
        for k in ("palm_lo", "palm_hi", "finger_x", "finger_z", "pad_z"):
            setattr(self, k, np.asarray(getattr(self, k), dtype=np.float64))
        if self.wrist_lo is not None:
            self.wrist_lo = np.asarray(self.wrist_lo, dtype=np.float64)
            self.wrist_hi = np.asarray(self.wrist_hi, dtype=np.float64)

    @classmethod
    def from_dict(cls, d: dict, source: str = "dict") -> "GripperGeometry":
        """Build from a loosely specified dict, using defaults for missing fields.

        Finger y is taken relative to the inner face unless finger_thickness is given.
        """
        g = cls()
        d = {str(k).lower(): v for k, v in d.items()}

        def box(v):
            if isinstance(v, dict):
                v = {str(k).lower(): x for k, x in v.items()}
                for a, b in (("lo", "hi"), ("min", "max"), ("lower", "upper"), ("bmin", "bmax")):
                    if a in v and b in v:
                        return np.asarray(v[a], float), np.asarray(v[b], float)
                if all(k in v for k in "xyz"):
                    arr = np.asarray([v["x"], v["y"], v["z"]], float)
                    return arr[:, 0], arr[:, 1]
                for k in ("box", "aabb", "bounds"):
                    if k in v:
                        return box(v[k])
                raise ValueError(f"unrecognised box {v}")
            arr = np.asarray(v, float)
            if arr.shape == (2, 3):
                return arr[0], arr[1]
            if arr.shape == (3, 2):
                return arr[:, 0], arr[:, 1]
            raise ValueError(f"unrecognised box {v}")

        for key in ("palm", "hand", "palm_box", "hand_box"):
            if key in d:
                g.palm_lo, g.palm_hi = box(d[key])
                break
        for key in ("wrist", "link7", "wrist_box"):
            if key in d:
                if d[key] is None:
                    g.wrist_lo = g.wrist_hi = None
                else:
                    g.wrist_lo, g.wrist_hi = box(d[key])
                break
        for key in ("finger", "finger_box", "finger_pos", "left_finger", "finger_plus"):
            if key in d:
                lo, hi = box(d[key])
                g.finger_x = np.array([lo[0], hi[0]])
                g.finger_z = np.array([lo[2], hi[2]])
                if "finger_thickness" not in d:
                    g.finger_offset = float(lo[1])
                    g.finger_thickness = float(hi[1] - lo[1])
                break
        for key in ("pad", "pads", "pad_box"):
            if key in d:
                v = d[key]
                try:
                    lo, hi = box(v)
                    g.pad_z = np.array([lo[2], hi[2]])
                except ValueError:
                    if isinstance(v, dict) and "z" in v:
                        g.pad_z = np.asarray(v["z"], float)
                break
        # flat keys override everything
        for k in ("palm_lo", "palm_hi", "wrist_lo", "wrist_hi", "finger_x", "finger_z", "pad_z"):
            if k in d and d[k] is not None:
                setattr(g, k, np.asarray(d[k], float))
        for k in ("finger_thickness", "finger_offset", "closing_x_scale", "max_width", "tcp_offset"):
            if k in d and np.isscalar(d[k]):
                setattr(g, k, float(d[k]))
        g.source = source
        g._validate()
        return g

    def _validate(self):
        assert np.all(self.palm_hi > self.palm_lo), "palm box inverted"
        assert self.finger_x[1] > self.finger_x[0] and self.finger_z[1] > self.finger_z[0]
        assert self.pad_z[1] > self.pad_z[0] and self.finger_thickness > 0

    @classmethod
    def load(cls, path: str | None = None) -> "GripperGeometry":
        """Load ``path`` (default: kinematics/panda_gripper_geometry.json) or use defaults."""
        path = GEOMETRY_JSON if path is None else path
        if path and os.path.isfile(path):
            try:
                with open(path) as f:
                    return cls.from_dict(json.load(f), source=path)
            except Exception as e:  # pragma: no cover - defensive
                warnings.warn(f"could not parse gripper geometry {path}: {e}; using defaults")
        return cls()

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}

    def symmetrized(self) -> "GripperGeometry":
        """Copy unchanged by Rz(pi) about the approach, so the canonicalisation flip cannot add a collision."""
        import dataclasses

        def sym(lo, hi, axes=(0, 1)):
            lo, hi = lo.copy(), hi.copy()
            for k in axes:
                lo[k], hi[k] = min(lo[k], -hi[k]), max(hi[k], -lo[k])
            return lo, hi

        g = dataclasses.replace(self)
        g.palm_lo, g.palm_hi = sym(self.palm_lo, self.palm_hi)
        if self.wrist_lo is not None:
            g.wrist_lo, g.wrist_hi = sym(self.wrist_lo, self.wrist_hi)
        fx = max(abs(self.finger_x[0]), abs(self.finger_x[1]))
        g.finger_x = np.array([-fx, fx])
        g.source = self.source + " (symmetrized)"
        return g

    def static_boxes(self, include_wrist: bool = True):
        """(lo (B,3), hi (B,3)) of width-independent boxes: palm [, wrist]."""
        lo, hi = [self.palm_lo], [self.palm_hi]
        if include_wrist and self.wrist_lo is not None:
            lo.append(self.wrist_lo)
            hi.append(self.wrist_hi)
        return np.stack(lo), np.stack(hi)

    def finger_boxes(self, width, y_shift=0.0):
        """Finger boxes for opening width (scalar or (G,)), optional TCP-frame y shift.

        Returns: lo, hi (...,2,3); index 0 = +y finger, 1 = -y finger.
        """
        w = np.asarray(width, dtype=np.float64)
        s = np.asarray(y_shift, dtype=np.float64)
        shape = np.broadcast(w, s).shape
        w = np.broadcast_to(w, shape)
        s = np.broadcast_to(s, shape)
        inner = 0.5 * w + self.finger_offset
        outer = inner + self.finger_thickness
        lo = np.empty(shape + (2, 3))
        hi = np.empty(shape + (2, 3))
        lo[..., :, 0], hi[..., :, 0] = self.finger_x[0], self.finger_x[1]
        lo[..., :, 2], hi[..., :, 2] = self.finger_z[0], self.finger_z[1]
        lo[..., 0, 1], hi[..., 0, 1] = s + inner, s + outer
        lo[..., 1, 1], hi[..., 1, 1] = s - outer, s - inner
        return lo, hi

    def boxes(self, width, include_wrist: bool = True, y_shift=0.0):
        """All boxes for opening ``width``: (...,B,3) lo, hi (static first, then fingers)."""
        flo, fhi = self.finger_boxes(width, y_shift)
        slo, shi = self.static_boxes(include_wrist)
        shape = flo.shape[:-2]
        slo = np.broadcast_to(slo, shape + slo.shape)
        shi = np.broadcast_to(shi, shape + shi.shape)
        return np.concatenate([slo, flo], axis=-2), np.concatenate([shi, fhi], axis=-2)

    def closing_box(self, width, y_shift=0.0, full_length: bool = False):
        """Closing region between the pads: x in finger_x*scale, |y - shift| < w/2, z in pad_z.

        full_length uses the whole finger z-range: the volume the fingers sweep while closing.
        """
        w = np.asarray(width, dtype=np.float64)
        s = np.asarray(y_shift, dtype=np.float64)
        shape = np.broadcast(w, s).shape
        lo = np.empty(shape + (3,))
        hi = np.empty(shape + (3,))
        lo[..., 0], hi[..., 0] = self.finger_x * self.closing_x_scale
        lo[..., 1], hi[..., 1] = s - 0.5 * w, s + 0.5 * w
        lo[..., 2], hi[..., 2] = self.finger_z if full_length else self.pad_z
        return lo, hi

    def corners(self, width=MAX_WIDTH, include_wrist: bool = True, fingers_only: bool = False) -> np.ndarray:
        """(B*8, 3) box corners in the TCP frame (used for the table check)."""
        lo, hi = self.finger_boxes(width) if fingers_only else self.boxes(width, include_wrist)
        bits = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=bool)  # (8,3)
        c = np.where(bits[None], hi[:, None, :], lo[:, None, :])     # (B,8,3)
        return c.reshape(-1, 3)

    def bounding_radius(self, width=MAX_WIDTH, include_wrist: bool = True) -> float:
        return float(np.linalg.norm(self.corners(width, include_wrist), axis=1).max())

    def spheres(self, width=MAX_WIDTH, include_wrist: bool = False, cell_ratio: float = 1.0):
        """Spheres (centres (S,3), radii (S,)) in the TCP frame whose union contains every gripper box."""
        lo, hi = self.boxes(width, include_wrist)
        C, R = [], []
        for l, h in zip(lo, hi):
            c0, half = 0.5 * (l + h), 0.5 * (h - l)
            order = np.argsort(half)
            a = half[order[0]]
            n = np.ones(3, dtype=int)
            for ax in order[1:]:
                n[ax] = max(1, int(np.ceil(half[ax] / (cell_ratio * a) - 1e-9)))
            cell = half / n
            grids = [c0[k] - half[k] + cell[k] * (2 * np.arange(n[k]) + 1) for k in range(3)]
            cen = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 3)
            C.append(cen)
            R.append(np.full(len(cen), np.linalg.norm(cell)))
        return np.concatenate(C), np.concatenate(R)


_GEOM_CACHE: dict = {}


def _mtime(path: str):
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def get_geometry(path: str | None = None, reload: bool = False) -> GripperGeometry:
    """Cached GripperGeometry.load; reloads when the JSON file appears, changes or goes away."""
    key = path or GEOMETRY_JSON
    stamp = _mtime(key)
    hit = _GEOM_CACHE.get(key)
    if reload or hit is None or hit[0] != stamp:
        hit = (stamp, GripperGeometry.load(key))
        _GEOM_CACHE[key] = hit
    return hit[1]


def _rotate_T(d: np.ndarray, R: np.ndarray) -> np.ndarray:
    """q = R^T d for d (...,3), R (...,3,3); explicit sums are faster than einsum and bit-identical for all callers."""
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    return np.stack([dx * R[..., 0, j] + dy * R[..., 1, j] + dz * R[..., 2, j] for j in range(3)], axis=-1)


def world_to_tcp(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """TCP-frame coordinates q = R^T (p - t). T (4,4)->(N,3); T (G,4,4)->(G,N,3)."""
    T = np.asarray(T, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    if T.ndim == 2:
        return (pts - T[:3, 3]) @ T[:3, :3]
    return _rotate_T(pts[None] - T[:, None, :3, 3], T[:, None, :3, :3])


def tcp_to_world(T: np.ndarray, q: np.ndarray) -> np.ndarray:
    """World coordinates p = R q + t for TCP-frame points q (N,3)."""
    T = np.asarray(T, dtype=np.float64)
    return np.asarray(q) @ T[:3, :3].T + T[:3, 3]


def points_in_boxes(q: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Point-in-any-box test: q (...,N,3), lo/hi (...,B,3) -> (...,N) bool."""
    out = np.zeros(q.shape[:-1], dtype=bool)
    for b in range(lo.shape[-2]):
        inb = np.ones(q.shape[:-1], dtype=bool)
        for k in range(3):
            qk = q[..., k]
            inb &= (qk >= lo[..., b, k, None]) & (qk <= hi[..., b, k, None])
        out |= inb
    return out


def boxes_reach(lo: np.ndarray, hi: np.ndarray) -> float:
    """Largest distance from the TCP to any point in boxes [lo, hi] (...,3); use on the inflated boxes."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if lo.size == 0:
        return 0.0
    return float(np.sqrt(np.maximum(lo * lo, hi * hi).sum(-1)).max())


def _prefilter(Ts, pts, radius):
    """Keep only points inside the AABB of all TCP positions inflated by ``radius``."""
    P = Ts[:, :3, 3]
    lo = P.min(0) - radius
    hi = P.max(0) + radius
    m = np.all((pts >= lo) & (pts <= hi), axis=1)
    return pts[m]


def _chunks(G, N, budget=2_000_000):
    step = max(1, int(budget // max(N, 1)))
    for s in range(0, G, step):
        yield slice(s, min(G, s + step))


def collides_batch(Ts, widths, pts, margin: float = 0.005, geom: GripperGeometry | None = None,
                   include_wrist: bool = True, sweep: float = 0.0, return_counts: bool = False):
    """Batched gripper-vs-point-cloud collision test.

    Ts: (G,4,4) TCP poses. widths: scalar or (G,). pts: (N,3) world. margin: box inflation (m).
    sweep: if > 0, also check the straight approach from T - sweep*approach to T.
    Returns: (G,) bool, or (G,) int counts of colliding points if return_counts.
    """
    geom = geom or get_geometry()
    Ts = np.asarray(Ts, dtype=np.float64).reshape(-1, 4, 4)
    G = Ts.shape[0]
    widths = np.broadcast_to(np.asarray(widths, dtype=np.float64), (G,))
    out = np.zeros(G, dtype=np.int64)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if G == 0 or pts.shape[0] == 0:
        return out if return_counts else out.astype(bool)
    lo, hi = geom.boxes(widths, include_wrist)       # (G,B,3)
    lo = lo - margin
    hi = hi + margin
    lo[..., 2] -= sweep
    pts = _prefilter(Ts, pts, boxes_reach(lo, hi))   # exact reach of the inflated, swept boxes
    if pts.shape[0] == 0:
        return out if return_counts else out.astype(bool)
    for sl in _chunks(G, pts.shape[0] * lo.shape[1]):
        q = world_to_tcp(Ts[sl], pts)                  # (g,N,3)
        hit = points_in_boxes(q, lo[sl], hi[sl])        # (g,N)
        out[sl] = hit.sum(1)
    return out if return_counts else out > 0


def collides(T, width, pts, margin: float = 0.005, geom: GripperGeometry | None = None,
             include_wrist: bool = True, sweep: float = 0.0) -> bool:
    """True if any point lies inside the (margin-inflated) palm / wrist / finger boxes."""
    return bool(collides_batch(np.asarray(T)[None], width, pts, margin, geom, include_wrist, sweep)[0])


def voxel_downsample(pts: np.ndarray, voxel: float, return_groups: bool = False):
    """Keep, per voxel, the member nearest the centroid and the radius of a ball covering the voxel.

    Returns: idx (V,) sorted indices into pts, radius (V,). With return_groups also members (N,),
    start (V,), count (V,): voxel v holds pts[members[start[v]:start[v] + count[v]]].
    """
    n = len(pts)
    if voxel <= 0 or n == 0:
        ar = np.arange(n)
        return (ar, np.zeros(n), ar, ar, np.ones(n, dtype=np.int64)) if return_groups else (ar, np.zeros(n))
    k = np.floor(pts / voxel).astype(np.int64)
    k -= k.min(0)
    dims = k.max(0) + 1
    key = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    nv = int(inv.max()) + 1
    cnt = np.bincount(inv, minlength=nv)
    cen = np.stack([np.bincount(inv, weights=pts[:, j], minlength=nv) for j in range(3)], 1) / cnt[:, None]
    dc = pts - cen[inv]
    order = np.lexsort((np.einsum("ij,ij->i", dc, dc), inv))     # by voxel, then distance to centroid
    first = np.flatnonzero(np.r_[True, inv[order][1:] != inv[order][:-1]])
    rep = order[first]                                           # (nv,) closest member per voxel
    dist = np.linalg.norm(pts[order] - pts[rep][inv[order]], axis=1)
    rad = np.maximum.reduceat(dist, first)
    s = np.argsort(rep)                                          # deterministic, sorted output
    if return_groups:
        return rep[s], rad[s], order, first[s], cnt[s]
    return rep[s], rad[s]


class CloudCollider:
    """Point cloud prepared for many exact gripper collision queries (same result as collides_batch).

    A broad phase on voxel representatives and covering radii leaves few voxels to test point by point.
    """

    def __init__(self, pts, voxel: float = 0.005):
        P = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        self.pts = P[np.all(np.isfinite(P), axis=1)]
        idx, rad, self.members, self.start, self.count = voxel_downsample(self.pts, voxel, return_groups=True)
        self.rep = self.pts[idx]
        self.rad = rad + 1e-9                    # guard against rounding in the bound

    def __len__(self):
        return len(self.pts)

    def collides(self, Ts, widths, margin: float = 0.005, geom: GripperGeometry | None = None,
                 include_wrist: bool = True, sweep: float = 0.0) -> np.ndarray:
        """(G,) bool, same semantics as :func:`collides_batch` (full-resolution cloud)."""
        geom = geom or get_geometry()
        Ts = np.asarray(Ts, dtype=np.float64).reshape(-1, 4, 4)
        G = Ts.shape[0]
        out = np.zeros(G, dtype=bool)
        if G == 0 or len(self.rep) == 0:
            return out
        widths = np.broadcast_to(np.asarray(widths, dtype=np.float64), (G,))
        lo, hi = geom.boxes(widths, include_wrist)          # (G,B,3)
        lo = lo - margin
        hi = hi + margin
        lo[..., 2] -= sweep
        P = Ts[:, :3, 3]
        reach = boxes_reach(lo, hi) + self.rad.max()
        near = np.flatnonzero(np.all((self.rep >= P.min(0) - reach) & (self.rep <= P.max(0) + reach), axis=1))
        if near.size == 0:
            return out
        rep, rad = self.rep[near], self.rad[near]
        # box centres / half-extents: L_inf outside-distance e = min_b max_k(|q_k - c_bk| - h_bk)
        bc, bh = 0.5 * (lo + hi), 0.5 * (hi - lo)                           # (G,B,3)
        pairs_g, pairs_v = [], []
        for sl in _chunks(G, near.size * lo.shape[1]):
            q = world_to_tcp(Ts[sl], rep)                                    # (g,V,3)
            e = np.full(q.shape[:2], np.inf)
            for b in range(bc.shape[1]):
                eb = np.abs(q[..., 0] - bc[sl, b, 0, None]) - bh[sl, b, 0, None]
                for k in (1, 2):
                    eb = np.maximum(eb, np.abs(q[..., k] - bc[sl, b, k, None]) - bh[sl, b, k, None])
                e = np.minimum(e, eb)
            sure = e < -1e-12                                                # rep itself collides
            out[sl] = sure.any(1)
            gi, vi = np.nonzero((e <= rad[None]) & ~out[sl][:, None])       # undecided voxels
            pairs_g.append(gi + sl.start)
            pairs_v.append(near[vi])
        gi, vi = np.concatenate(pairs_g), np.concatenate(pairs_v)
        if gi.size == 0:
            return out
        # narrow phase: test every member point of the undecided voxels
        cnt = self.count[vi]
        tot = int(cnt.sum())
        off = np.repeat(self.start[vi] - np.cumsum(cnt) + cnt, cnt) + np.arange(tot)
        mem = self.members[off]
        g = np.repeat(gi, cnt)
        for s0 in range(0, tot, 200_000):
            s1 = min(tot, s0 + 200_000)
            gg = g[s0:s1]
            q = _rotate_T(self.pts[mem[s0:s1]] - Ts[gg, :3, 3], Ts[gg, :3, :3])
            hit = points_in_boxes(q[:, None, :], lo[gg], hi[gg])[:, 0]
            out[np.unique(gg[hit])] = True
        return out


def closing_region_mask_batch(Ts, widths, pts, geom: GripperGeometry | None = None,
                              full_length: bool = False) -> np.ndarray:
    """(G,N) bool: points in the closing region of each grasp (whole finger length if full_length)."""
    geom = geom or get_geometry()
    Ts = np.asarray(Ts, dtype=np.float64).reshape(-1, 4, 4)
    G = Ts.shape[0]
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    widths = np.broadcast_to(np.asarray(widths, dtype=np.float64), (G,))
    lo, hi = geom.closing_box(widths, full_length=full_length)          # (G,3)
    out = np.zeros((G, pts.shape[0]), dtype=bool)
    for sl in _chunks(G, pts.shape[0]):
        q = world_to_tcp(Ts[sl], pts)
        out[sl] = np.all((q >= lo[sl, None]) & (q <= hi[sl, None]), axis=-1)
    return out


def closing_region_mask(T, width, pts, geom: GripperGeometry | None = None,
                        full_length: bool = False) -> np.ndarray:
    """(N,) bool: points in the closing region of one grasp (see closing_box)."""
    return closing_region_mask_batch(np.asarray(T)[None], width, pts, geom, full_length)[0]


def voxel_labels(pts: np.ndarray, voxel: float) -> np.ndarray:
    """(N,) dense int voxel id per point, on the same grid as voxel_downsample."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0 or voxel <= 0:                  # no reduction: every point is its own voxel
        return np.arange(len(pts), dtype=np.int64)
    k = np.floor(pts / voxel).astype(np.int64)
    k -= k.min(0)
    dims = k.max(0) + 1
    return np.unique((k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2], return_inverse=True)[1].ravel()


def closing_volume_voxel_counts(Ts, widths, pts, labels, geom: GripperGeometry | None = None) -> np.ndarray:
    """(G,) number of distinct voxels (labels of pts) with a point between the closing fingers.

    Counting voxels, not points, keeps the tolerance independent of cloud density.
    """
    geom = geom or get_geometry()
    Ts = np.asarray(Ts, dtype=np.float64).reshape(-1, 4, 4)
    G = Ts.shape[0]
    out = np.zeros(G, dtype=np.int64)
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if G == 0 or len(pts) == 0:
        return out
    widths = np.broadcast_to(np.asarray(widths, dtype=np.float64), (G,))
    lo, hi = geom.closing_box(widths, full_length=True)                 # (G,3)
    keep = np.all((pts >= Ts[:, :3, 3].min(0) - boxes_reach(lo, hi))
                  & (pts <= Ts[:, :3, 3].max(0) + boxes_reach(lo, hi)), axis=1)
    pts, labels = pts[keep], np.asarray(labels)[keep]
    if len(pts) == 0:
        return out
    for sl in _chunks(G, pts.shape[0]):
        q = world_to_tcp(Ts[sl], pts)                                     # (g,N,3)
        inside = np.all((q >= lo[sl, None]) & (q <= hi[sl, None]), axis=-1)
        for i, m in zip(range(sl.start, sl.stop), inside):
            out[i] = np.unique(labels[m]).size
    return out


# old name, kept for callers
closing_region_points = closing_region_mask


def lowest_z_batch(Ts, widths=MAX_WIDTH, geom: GripperGeometry | None = None,
                   include_wrist: bool = True, fingers_only: bool = False) -> np.ndarray:
    """World z of the lowest gripper box corner for each pose (G,) — for table checks."""
    geom = geom or get_geometry()
    Ts = np.asarray(Ts, dtype=np.float64).reshape(-1, 4, 4)
    G = Ts.shape[0]
    widths = np.broadcast_to(np.asarray(widths, dtype=np.float64), (G,))
    out = np.empty(G)
    for w in np.unique(widths):
        sel = widths == w
        C = geom.corners(w, include_wrist, fingers_only)
        z = np.einsum("gk,nk->gn", Ts[sel, 2, :3], C) + Ts[sel, 2, 3][:, None]
        out[sel] = z.min(1)
    return out


def lowest_z(T, width=MAX_WIDTH, geom: GripperGeometry | None = None, fingers_only: bool = False) -> float:
    return float(lowest_z_batch(np.asarray(T)[None], width, geom, fingers_only=fingers_only)[0])


def gripper_spheres(T=None, width=MAX_WIDTH, geom: GripperGeometry | None = None,
                    include_wrist: bool = False):
    """Sphere cover of the open gripper: centres (S,3), radii (S,); TCP frame if T is None, else world."""
    geom = geom or get_geometry()
    c, r = geom.spheres(width, include_wrist)
    if T is not None:
        c = tcp_to_world(T, c)
    return c, r
