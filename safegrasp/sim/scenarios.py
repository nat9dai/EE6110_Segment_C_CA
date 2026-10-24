"""Generate seeded scene specs (pure numpy): which GSO objects go where before physics settles.

Kinds: sparse, medium, dense, and 'hidden' (small objects hidden behind a tall target).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# name -> (n objects, drop-patch half extents (x, y) [m], overlap factor)
DENSITY = {
    "sparse": (4, (0.14, 0.20), 1.0),
    "medium": (8, (0.13, 0.17), 0.85),
    "dense": (12, (0.11, 0.14), 0.7),
}
WS_CENTER = np.array([0.55, 0.0])
MAIN_CAM_EYE = np.array([1.20, 0.0, 0.33])


@dataclass
class ObjPlacement:
    slot: int                 # index into the object pool
    pos: np.ndarray           # (3,) world drop position of the bbox centre [m]
    quat: np.ndarray          # (4,) (w,x,y,z)
    role: str = "clutter"     # clutter | occluder | hidden


@dataclass
class SceneSpec:
    seed: int
    kind: str
    placements: list[ObjPlacement]
    target_slot: int | None = None     # None: chosen after settling
    meta: dict = field(default_factory=dict)


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def random_quat(rng) -> np.ndarray:
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def _footprint_radius(ext) -> float:
    return 0.5 * float(np.hypot(ext[0], ext[1]))


def _sample_layout(rng, extents, n, half, overlap, center=WS_CENTER, upright_p=0.7, max_tries=200):
    """Rejection-sample n drop poses as a list of (pos, quat); ``extents`` is (n,3) per-object xyz."""
    placed = []  # (xy, radius)
    out = []
    for i in range(n):
        ext = np.asarray(extents[i])
        upright = rng.random() < upright_p
        yaw = rng.uniform(-np.pi, np.pi)
        if upright:
            quat = quat_from_yaw(yaw)
            r = _footprint_radius(ext)
            h = 0.5 * ext[2]
        else:
            quat = random_quat(rng)
            r = 0.5 * float(np.max(ext))
            h = 0.5 * float(np.linalg.norm(ext))
        best = None
        for _ in range(max_tries):
            xy = center + rng.uniform(-1, 1, size=2) * np.asarray(half)
            if all(np.linalg.norm(xy - p) >= overlap * (r + rp) for p, rp in placed):
                best = xy
                break
        if best is None:  # accept overlap: the object drops onto a pile
            best = center + rng.uniform(-1, 1, size=2) * np.asarray(half)
        placed.append((best, r))
        z = h + 0.005 + 0.05 * i * (overlap < 1.0)  # stagger heights in denser scenes
        out.append((np.array([best[0], best[1], z]), quat))
    return out


def make_scene(seed: int, kind: str, pool_extents: np.ndarray, pool_ok: np.ndarray | None = None) -> SceneSpec:
    """Build a scene spec; ``pool_extents`` is (K,3) object-frame xyz extents of the pool [m]."""
    rng = np.random.default_rng(seed)
    K = len(pool_extents)
    avail = np.arange(K) if pool_ok is None else np.flatnonzero(pool_ok)
    if kind in DENSITY:
        n, half, overlap = DENSITY[kind]
        slots = rng.choice(avail, size=min(n, len(avail)), replace=False)
        layout = _sample_layout(rng, pool_extents[slots], len(slots), half, overlap)
        pl = [ObjPlacement(int(s), p, q) for s, (p, q) in zip(slots, layout)]
        return SceneSpec(seed, kind, pl)
    if kind == "hidden":
        return _make_hidden(rng, seed, pool_extents, avail)
    raise ValueError(kind)


def _half_along(ext, yaw, u):
    """Half extent of an upright box (ext x,y) with yaw, along unit xy direction u."""
    c, s_ = np.cos(yaw), np.sin(yaw)
    ax, ay = np.array([c, s_]), np.array([-s_, c])
    return 0.5 * (ext[0] * abs(ax @ u) + ext[1] * abs(ay @ u))


def _make_hidden(rng, seed, ext, avail) -> SceneSpec:
    """Place a tall occluder (the target), 1-2 short objects in its main-camera shadow, and clutter."""
    height = ext[:, 2]
    wide = np.maximum(ext[:, 0], ext[:, 1])
    narrow = np.minimum(ext[:, 0], ext[:, 1])
    foot = np.array([_footprint_radius(e) for e in ext])
    occ_cand = [s for s in avail if 0.09 <= height[s] <= 0.22 and narrow[s] <= 0.065 and wide[s] >= 0.06 and foot[s] <= 0.10]
    if not occ_cand:
        occ_cand = list(avail[np.argsort(-height[avail])[:5]])
    occ = int(rng.choice(occ_cand))
    p_occ = WS_CENTER + rng.uniform([-0.05, -0.10], [0.07, 0.10])
    # the ray over the occluder's top drops tan(elev) per metre: h_max = H - tan(elev) * (depth + gap)
    dist = np.linalg.norm(p_occ - MAIN_CAM_EYE[:2])
    tan_el = (MAIN_CAM_EYE[2] - height[occ]) / max(dist, 1e-3)
    def fits(s):
        return height[s] <= height[occ] - tan_el * (narrow[s] + 0.02) - 0.01 and wide[s] <= 0.95 * wide[occ]
    fit = [s for s in avail if s != occ and fits(s)]
    # prefer the tallest that fit: they reach into the fingers' sweep
    hid_cand = sorted(fit, key=lambda s: -height[s])[:4]
    if not hid_cand:
        hid_cand = [s for s in avail if s != occ and fits(s)]
    if not hid_cand:
        hid_cand = [s for s in avail if s != occ and height[s] <= 0.6 * height[occ]] or [s for s in avail if s != occ]
    n_hidden = 1 if rng.random() < 0.6 else 2
    hidden = [int(s) for s in rng.choice(hid_cand, size=min(n_hidden, len(hid_cand)), replace=False)]

    u = p_occ - MAIN_CAM_EYE[:2]
    u /= np.linalg.norm(u)
    v = np.array([-u[1], u[0]])
    # wide face toward the camera: longer horizontal axis along v
    yaw_u = np.arctan2(u[1], u[0])
    yaw = yaw_u + (np.pi / 2 if ext[occ, 0] >= ext[occ, 1] else 0.0) + rng.uniform(-0.15, 0.15)
    pl = [ObjPlacement(occ, np.array([p_occ[0], p_occ[1], 0.5 * ext[occ, 2] + 0.003]), quat_from_yaw(yaw), "occluder")]
    d_occ = _half_along(ext[occ], yaw, u)
    w_occ = _half_along(ext[occ], yaw, v)
    for j, h in enumerate(hidden):
        yh = yaw_u + (np.pi / 2 if ext[h, 0] >= ext[h, 1] else 0.0) + rng.uniform(-0.3, 0.3)
        d_h = _half_along(ext[h], yh, u)
        w_h = _half_along(ext[h], yh, v)
        gap = d_occ + d_h + rng.uniform(0.002, 0.008)
        lat_max = max(0.0, w_occ - w_h)
        lateral = rng.uniform(-lat_max, lat_max) if len(hidden) == 1 else (-1) ** j * min(lat_max, w_h + 0.004)
        xy = p_occ + gap * u + lateral * v
        pl.append(ObjPlacement(h, np.array([xy[0], xy[1], 0.5 * ext[h, 2] + 0.003]), quat_from_yaw(yh), "hidden"))
    rest = [s for s in avail if s != occ and s not in hidden]
    n_cl = int(rng.integers(3, 6))
    cl = rng.choice(rest, size=min(n_cl, len(rest)), replace=False)
    taken = [(p.pos[:2], foot[p.slot]) for p in pl]
    for s in cl:
        for _ in range(200):
            xy = WS_CENTER + rng.uniform(-1, 1, size=2) * np.array([0.14, 0.20])
            if all(np.linalg.norm(xy - p) >= 1.05 * (foot[s] + r) for p, r in taken):
                break
        taken.append((xy, foot[s]))
        pl.append(ObjPlacement(int(s), np.array([xy[0], xy[1], 0.5 * ext[s, 2] + 0.005]), quat_from_yaw(rng.uniform(-np.pi, np.pi))))
    return SceneSpec(seed, "hidden", pl, target_slot=occ, meta={"occluder": occ, "hidden": hidden})
