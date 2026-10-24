"""Depth-image utilities in numpy / scipy: sensor noise, back-projection and normal estimation.
Depth is camera-frame z, so p_c = z K^{-1} [u, v, 1]^T and p_w = R_wc p_c + t_wc."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .types import SEG_ROBOT, View

__all__ = [
    "add_depth_noise",
    "backproject",
    "backproject_views",
    "estimate_normals",
    "depth_edges",
    "pixel_rays",
    "transform_points",
]


def depth_edges(depth: np.ndarray, thresh: float = 0.02) -> np.ndarray:
    """(H,W) bool: valid pixels whose 4-neighbour differs in depth by > thresh [m].
    Both pixels of the pair are flagged, as flying pixels appear on both sides of an edge."""
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    d = np.where(valid, d, 0.0)                       # avoid inf - inf
    edge = np.zeros(d.shape, dtype=bool)
    jump = valid[:, 1:] & valid[:, :-1] & (np.abs(d[:, 1:] - d[:, :-1]) > thresh)
    edge[:, 1:] |= jump
    edge[:, :-1] |= jump
    jump = valid[1:, :] & valid[:-1, :] & (np.abs(d[1:, :] - d[:-1, :]) > thresh)
    edge[1:, :] |= jump
    edge[:-1, :] |= jump
    return edge


def add_depth_noise(depth, rng, sigma0=0.001, sigma2=0.002, dropout=0.0, edge_dropout=True,
                    edge_thresh=0.02, edge_prob=1.0) -> np.ndarray:
    """Add RGB-D sensor artefacts to a clean depth image; returns a new float32 array (invalid = 0).

    Axial noise sigma(z) = sigma0 + sigma2 z^2 [m], random dropout with probability ``dropout``,
    and dropout of edge pixels (jumps > edge_thresh on the clean image) with probability edge_prob.
    """
    d = np.array(depth, dtype=np.float32, copy=True)
    valid = np.isfinite(d) & (d > 0)
    z = np.where(valid, d, 0.0).astype(np.float64)

    invalid = ~valid
    if edge_dropout:
        edge = depth_edges(d, edge_thresh)
        if edge_prob < 1.0:
            edge &= rng.random(d.shape) < edge_prob
        invalid |= edge

    sigma = sigma0 + sigma2 * z * z
    z = z + sigma * rng.standard_normal(d.shape)

    if dropout > 0.0:
        invalid |= rng.random(d.shape) < dropout

    invalid |= z <= 0.0
    out = z.astype(np.float32)
    out[invalid] = 0.0
    return out


_RAY_CACHE: dict = {}


def pixel_rays(K, H: int, W: int) -> np.ndarray:
    """(H,W,3) camera-frame rays K^{-1} [u, v, 1]^T with unit z, so p_c = depth * ray.
    Cached per (K, H, W); treat as read-only."""
    K = np.asarray(K, dtype=np.float64)
    key = (K.tobytes(), int(H), int(W))
    r = _RAY_CACHE.get(key)
    if r is None:
        Kinv = np.linalg.inv(K)
        u = np.arange(W, dtype=np.float64)[None, :, None]
        v = np.arange(H, dtype=np.float64)[:, None, None]
        r = u * Kinv[:, 0] + v * Kinv[:, 1] + Kinv[:, 2]
        r.flags.writeable = False
        if len(_RAY_CACHE) >= 16:
            _RAY_CACHE.clear()
        _RAY_CACHE[key] = r
    return r


def transform_points(T, pts) -> np.ndarray:
    """``p' = R p + t`` for (N,3) points (einsum: avoids multithreaded BLAS for skinny matmuls)."""
    T = np.asarray(T, dtype=np.float64)
    return np.einsum("ij,nj->ni", T[:3, :3], pts) + T[:3, 3]


def backproject(view: View, stride: int = 1, include_robot: bool = False,
                max_depth: float = np.inf):
    """Back-project the valid pixels of a view (on a ``stride`` grid) to world points.

    Returns pts (N,3) world [m], obj_ids (N,) int32 (>= 0 object, -1 background, -2 robot) and
    pix (N,2) int64 (u, v) = (column, row). Robot pixels are skipped unless include_robot.
    """
    stride = max(int(stride), 1)
    depth = view.depth[::stride, ::stride]
    seg = view.seg[::stride, ::stride]
    valid = np.isfinite(depth) & (depth > 0) & (depth < max_depth)
    if not include_robot:
        valid &= seg != SEG_ROBOT
    rows, cols = np.nonzero(valid)
    z = depth[rows, cols].astype(np.float64)
    rays = pixel_rays(view.K, *view.depth.shape)[::stride, ::stride][rows, cols]   # (N,3)
    pts = transform_points(view.T_wc, rays * z[:, None])
    pix = np.stack([cols.astype(np.int64) * stride, rows.astype(np.int64) * stride], axis=1)
    return pts, seg[rows, cols].astype(np.int32), pix


def backproject_views(views, stride: int = 1, include_robot: bool = False):
    """Merge views into one cloud: (pts (N,3), obj_ids (N,), viewpoints (N,3)).
    viewpoints[i] is the centre of the camera that saw point i (for orienting normals)."""
    P, I, V = [], [], []
    for view in views:
        p, ids, _ = backproject(view, stride, include_robot)
        P.append(p)
        I.append(ids)
        V.append(np.broadcast_to(view.T_wc[:3, 3], p.shape))
    if not P:
        return np.zeros((0, 3)), np.zeros(0, np.int32), np.zeros((0, 3))
    return np.concatenate(P), np.concatenate(I), np.concatenate(V).copy()


def estimate_normals(pts, k: int = 16, viewpoint=None, tree: cKDTree | None = None) -> np.ndarray:
    """(N,3) unit surface normals by k-NN PCA, flipped to point toward the viewpoint.

    viewpoint: None (orient toward +z), (3,), or (N,3) per-point camera centres for merged clouds.
    Collinear or coincident neighbourhoods fall back to the reference direction (viewpoint - p or
    +z); non-finite points get a zero normal. tree: optional prebuilt cKDTree(pts).
    """
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    N = len(pts)
    normals = np.zeros((N, 3))
    if N == 0:
        return normals
    vp = None if viewpoint is None else np.asarray(viewpoint, dtype=np.float64)
    if vp is not None and vp.ndim > 1:
        vp = np.broadcast_to(vp.reshape(-1, 3), pts.shape)
    fin = np.all(np.isfinite(pts), axis=1)
    if not fin.all():                                 # drop NaN/inf points (cKDTree rejects them)
        normals[fin] = estimate_normals(pts[fin], k, None if vp is None else
                                        (vp if vp.ndim == 1 else vp[fin]))
        return normals

    if vp is None:
        ref = np.broadcast_to(np.array([0.0, 0.0, 1.0]), pts.shape)
    else:
        ref = vp - pts
    k = int(min(k, N))
    if k < 3:
        normals = np.array(ref, dtype=np.float64)
    else:
        if tree is None:
            tree = cKDTree(pts)
        _, nn = tree.query(pts, k=k, workers=-1)
        nb = pts[nn]                                  # (N,k,3)
        X = nb - nb.mean(axis=1, keepdims=True)
        C = np.einsum("nki,nkj->nij", X, X)           # (N,3,3) scatter matrices
        lam, V = np.linalg.eigh(C)                    # ascending eigenvalues
        normals = V[:, :, 0].copy()
        # degenerate neighbourhoods: the smallest-eigenvalue direction is arbitrary
        line = lam[:, 1] <= 1e-4 * lam[:, 2]          # collinear: normal is any dir. _|_ line
        if line.any():
            a = V[line, :, 2]                         # line direction
            r = ref[line]
            nl = r - np.einsum("ni,ni->n", r, a)[:, None] * a
            ok = np.linalg.norm(nl, axis=1) > 1e-9    # r parallel to the line: keep PCA normal
            normals[np.flatnonzero(line)[ok]] = nl[ok]
        point = lam[:, 2] <= 1e-16                    # coincident points: no geometry at all
        normals[point] = ref[point]

    # orientation: n . r > 0
    s = np.einsum("ni,ni->n", normals, ref)
    normals[s < 0] *= -1.0
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(nrm > 1e-12, normals / np.maximum(nrm, 1e-12), [0.0, 0.0, 1.0])
    return normals
