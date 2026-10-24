"""Grasp type and pose helpers shared by all proposers, rankers and executors.

Grasp.T: (4,4) TCP pose in world; col 1 = closing axis, col 2 = approach (into the object).
TCP = panda_hand origin + TCP_OFFSET along hand z. width = gap between finger pads (m).
Quaternions are (w, x, y, z).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TCP_OFFSET = 0.1034          # panda_hand origin -> TCP (m) along hand z
MAX_WIDTH = 0.08             # max finger-pad separation (m)
TOPDOWN_QUAT = (0.0, 1.0, 0.0, 0.0)   # (w,x,y,z): hand z = -world z, hand x = world x
DEFAULT_PREGRASP_DIST = 0.12


@dataclass
class Grasp:
    """A parallel-jaw grasp candidate.

    T: (4,4) TCP pose in world. width: opening to command (m), <= 0.08.
    score: higher is better. info: extra diagnostics (floats, JSON-safe).
    """
    T: np.ndarray
    width: float
    score: float
    method: str
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.T = np.asarray(self.T, dtype=np.float64).reshape(4, 4)
        self.width = float(self.width)
        self.score = float(self.score)

    @property
    def position(self) -> np.ndarray:
        """TCP position (3,)."""
        return self.T[:3, 3]

    @property
    def approach(self) -> np.ndarray:
        """Unit approach axis (hand z) (3,)."""
        return self.T[:3, 2]

    @property
    def closing(self) -> np.ndarray:
        """Unit closing axis (hand y) (3,)."""
        return self.T[:3, 1]

    @property
    def quat(self) -> np.ndarray:
        """TCP orientation as quaternion (w,x,y,z)."""
        return quat_from_R(self.T[:3, :3])

    def hand_pose(self) -> np.ndarray:
        """``panda_hand`` body pose that realises this TCP pose."""
        return hand_from_tcp(self.T)

    def copy(self) -> "Grasp":
        return Grasp(self.T.copy(), self.width, self.score, self.method, dict(self.info))


def _as_T(g) -> np.ndarray:
    return g.T if isinstance(g, Grasp) else np.asarray(g, dtype=np.float64)


def pregrasp_pose(g, dist: float = DEFAULT_PREGRASP_DIST) -> np.ndarray:
    """Pre-grasp TCP pose: T moved back by dist (m) along the approach. g: Grasp, (4,4) or (G,4,4)."""
    T = _as_T(g).copy()
    T[..., :3, 3] -= dist * T[..., :3, 2]
    return T


def hand_from_tcp(T_tcp: np.ndarray) -> np.ndarray:
    """T_world_hand = T_world_tcp @ Trans(0,0,-TCP_OFFSET) (works on (...,4,4))."""
    T = np.array(T_tcp, dtype=np.float64, copy=True)
    T[..., :3, 3] -= TCP_OFFSET * T[..., :3, 2]
    return T


def tcp_from_hand(T_hand: np.ndarray) -> np.ndarray:
    """T_world_tcp = T_world_hand @ Trans(0,0,+TCP_OFFSET) (works on (...,4,4))."""
    T = np.array(T_hand, dtype=np.float64, copy=True)
    T[..., :3, 3] += TCP_OFFSET * T[..., :3, 2]
    return T


def make_pose(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Assemble homogeneous transform(s) from rotation(s) (...,3,3) and position(s) (...,3)."""
    R = np.asarray(R, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.0
    return T


def frame_from_axes(closing: np.ndarray, approach: np.ndarray) -> np.ndarray:
    """Rotation(s) with col 1 = closing (made orthogonal to approach), col 2 = approach. (3,) or (N,3)."""
    a = np.asarray(approach, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    y = np.asarray(closing, dtype=np.float64)
    y = y - np.sum(y * a, axis=-1, keepdims=True) * a
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    x = np.cross(y, a)
    return np.stack([x, y, a], axis=-1)


def canonicalize_about_approach(T: np.ndarray, base_xy=(0.0, 0.0)) -> np.ndarray:
    """Pick the one of {T, T Rz(pi)} whose hand x points away from the base. (4,4) or (G,4,4).

    Both are the same physical grasp; this choice keeps joint 7 away from its limits.
    """
    T = np.array(T, dtype=np.float64, copy=True)
    single = T.ndim == 2
    if single:
        T = T[None]
    r = T[:, :2, 3] - np.asarray(base_xy, dtype=np.float64)[None]
    rn = np.linalg.norm(r, axis=1, keepdims=True)
    r = np.where(rn > 1e-6, r / np.maximum(rn, 1e-9), np.array([[1.0, 0.0]]))
    xdot = np.sum(T[:, :2, 0] * r, axis=1)
    # tie-breaks (x almost perpendicular to r): prefer x with +z component, then counter-clockwise x
    ccw = r[:, 0] * T[:, 1, 0] - r[:, 1] * T[:, 0, 0]
    key = np.where(np.abs(xdot) >= 1e-3, xdot, np.where(np.abs(T[:, 2, 0]) >= 1e-3, T[:, 2, 0], ccw))
    flip = key < 0
    T[flip, :3, 0] *= -1.0
    T[flip, :3, 1] *= -1.0
    return T[0] if single else T


def rotation_angle_between(R1: np.ndarray, R2: np.ndarray, symmetric: bool = True) -> np.ndarray:
    """Angle (rad) between rotations; with symmetric, the smaller of R2 and R2 Rz(pi). Broadcasts."""
    R1 = np.asarray(R1)
    R2 = np.asarray(R2)
    tr = np.einsum("...ij,...ij->...", R1, R2)
    ang = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    if symmetric:
        # R2*Rz(pi) negates columns 0 and 1 -> trace term changes sign on those columns
        c0 = np.einsum("...i,...i->...", R1[..., :, 0], R2[..., :, 0])
        c1 = np.einsum("...i,...i->...", R1[..., :, 1], R2[..., :, 1])
        c2 = np.einsum("...i,...i->...", R1[..., :, 2], R2[..., :, 2])
        tr2 = -c0 - c1 + c2
        ang2 = np.arccos(np.clip((tr2 - 1.0) / 2.0, -1.0, 1.0))
        ang = np.minimum(ang, ang2)
    return ang


def quat_from_R(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (3,3) -> unit quaternion (w,x,y,z) with w >= 0 (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def R_from_quat(q) -> np.ndarray:
    """Unit quaternion (w,x,y,z) -> rotation matrix (3,3)."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pose_to_pos_quat(T: np.ndarray):
    """(4,4) -> (pos (3,), quat (w,x,y,z) (4,))."""
    T = np.asarray(T, dtype=np.float64)
    return T[:3, 3].copy(), quat_from_R(T[:3, :3])


def pos_quat_to_pose(pos, quat) -> np.ndarray:
    return make_pose(R_from_quat(quat), np.asarray(pos, dtype=np.float64))


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def to_dict(g: Grasp) -> dict:
    """JSON-serialisable dict (T as nested list + pos/quat for readability)."""
    pos, quat = pose_to_pos_quat(g.T)
    return {
        "T": g.T.tolist(),
        "pos": pos.tolist(),
        "quat_wxyz": quat.tolist(),
        "width": float(g.width),
        "score": float(g.score),
        "method": str(g.method),
        "info": {str(k): _jsonable(v) for k, v in g.info.items()},
    }


def from_dict(d: dict) -> Grasp:
    """Inverse of :func:`to_dict`."""
    T = np.asarray(d["T"], dtype=np.float64) if "T" in d else pos_quat_to_pose(d["pos"], d["quat_wxyz"])
    return Grasp(T=T, width=d["width"], score=d["score"], method=d["method"], info=dict(d.get("info", {})))


def sort_grasps(grasps: list[Grasp]) -> list[Grasp]:
    """Stable sort, best (highest score) first."""
    return sorted(grasps, key=lambda g: -g.score)


def stack_poses(grasps: list[Grasp]) -> np.ndarray:
    """(G,4,4) array of TCP poses."""
    if len(grasps) == 0:
        return np.zeros((0, 4, 4))
    return np.stack([g.T for g in grasps])


def dedup_grasps(grasps: list[Grasp], pos_tol: float = 0.01, ang_tol: float = np.deg2rad(15.0)) -> list[Grasp]:
    """Drop a grasp if a better one is within pos_tol (m) and ang_tol (rad); input sorted best first."""
    if len(grasps) <= 1:
        return list(grasps)
    Ts = stack_poses(grasps)
    P = Ts[:, :3, 3]
    R = Ts[:, :3, :3]
    keep_idx: list[int] = []
    suppressed = np.zeros(len(grasps), dtype=bool)
    for i in range(len(grasps)):
        if suppressed[i]:
            continue
        keep_idx.append(i)
        rest = np.arange(i + 1, len(grasps))
        rest = rest[~suppressed[rest]]
        if rest.size == 0:
            continue
        close = np.linalg.norm(P[rest] - P[i], axis=1) < pos_tol
        if np.any(close):
            cand = rest[close]
            ang = rotation_angle_between(R[i][None], R[cand], symmetric=True)
            suppressed[cand[ang < ang_tol]] = True
    return [grasps[i] for i in keep_idx]


def make_cfg(cls, cfg=None, **overrides):
    """Build a config from None (defaults), a dict (unknown keys raise TypeError) or a dataclass."""
    import dataclasses
    if cfg is None:
        out = cls()
    elif isinstance(cfg, cls):
        out = dataclasses.replace(cfg)
    elif isinstance(cfg, dict):
        out = cls(**cfg)
    elif dataclasses.is_dataclass(cfg):
        names = {f.name for f in dataclasses.fields(cls)}
        out = cls(**{k: v for k, v in dataclasses.asdict(cfg).items() if k in names})
    else:
        raise TypeError(f"cannot build {cls.__name__} from {type(cfg)}")
    return dataclasses.replace(out, **overrides) if overrides else out
