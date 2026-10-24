"""Franka Panda kinematics in numpy (FK, Jacobians, collision spheres, IK) in Isaac Lab's body
frames, with world = panda_link0. Matches Isaac to < 1e-6 m (scripts/verify_fk_isaac.py)."""
from __future__ import annotations

import json
import os

import numpy as np

LINK_NAMES = ["panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4",
              "panda_link5", "panda_link6", "panda_link7", "panda_hand",
              "panda_leftfinger", "panda_rightfinger"]
LINK_INDEX = {n: i for i, n in enumerate(LINK_NAMES)}
HAND_ID, LEFT_ID, RIGHT_ID = 8, 9, 10

TCP_OFFSET = 0.1034      # panda_hand origin -> TCP along hand z  [m]
FINGER_Z = 0.0584        # panda_hand origin -> finger body origins along hand z  [m]
FLANGE_Z = 0.107         # panda_link7 origin -> panda_hand origin along link7 z  [m]
MAX_WIDTH = 0.08         # max opening (2 x 0.04)  [m]

# URDF joint origins (parent frame) and the fixed x-roll before each revolute joint
_JOINT_XYZ = np.array([[0.0, 0.0, 0.333],
                       [0.0, 0.0, 0.0],
                       [0.0, -0.316, 0.0],
                       [0.0825, 0.0, 0.0],
                       [-0.0825, 0.384, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.088, 0.0, 0.0]])
_JOINT_RX = np.array([0.0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2])

# joint limits [rad] as reported by Isaac (same as the URDF)
Q_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
QD_MAX_OFFICIAL = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])

# ready pose: TCP at (0.45, 0, 0.45) m pointing down, hand x = world x
Q_HOME = np.array([0.0, -0.3497, 0.0, -2.1264, 0.0, 1.7767, 0.7854])

_SPHERE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "panda_spheres.json")


# quaternions are (w, x, y, z)
def _rx(a: float) -> np.ndarray:
    c, s = np.round(np.cos(a), 15), np.round(np.sin(a), 15)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotz(a: float) -> np.ndarray:
    """3x3 rotation about z."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_to_matrix(q) -> np.ndarray:
    """(w,x,y,z) quaternion(s) (...,4) -> rotation matrix (...,3,3)."""
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def matrix_to_quat(R) -> np.ndarray:
    """Rotation matrix (...,3,3) -> (w,x,y,z) quaternion (...,4) with w >= 0 (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    if R.ndim > 2:
        out = [matrix_to_quat(r) for r in R.reshape(-1, 3, 3)]
        return np.asarray(out, dtype=float).reshape(R.shape[:-2] + (4,))
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(1.0 + tr)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    return q * (1.0 if q[0] >= 0 else -1.0)


def pose_to_matrix(pos, quat) -> np.ndarray:
    """(3,) position + (w,x,y,z) quaternion -> 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(quat)
    T[:3, 3] = pos
    return T


def rot_log(R: np.ndarray) -> np.ndarray:
    """SO(3) logarithm: rotation vector theta*a with R = exp([theta a]x), robust near theta = pi."""
    cos = min(1.0, max(-1.0, 0.5 * (R[0, 0] + R[1, 1] + R[2, 2] - 1.0)))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])  # = 2 sin(th) a
    if cos > -0.99:
        th = np.arccos(cos)
        return v * (0.5 if th < 1e-7 else 0.5 * th / np.sin(th))
    # near pi: symmetric part (R + R^T)/2 - cos I = (1 - cos) a a^T
    th = np.arccos(cos)
    B = 0.5 * (R + R.T) - cos * np.eye(3)
    i = int(np.argmax(np.diag(B)))
    a = B[:, i] / np.sqrt(max(B[i, i], 1e-12) * (1.0 - cos))
    a /= np.linalg.norm(a)
    if a @ v < 0:
        a = -a
    return th * a


def pose_error(T_cur: np.ndarray, T_goal: np.ndarray) -> np.ndarray:
    """6-vector [p_goal - p_cur ; log(R_goal R_cur^T)] in the world frame."""
    return np.concatenate([T_goal[:3, 3] - T_cur[:3, 3], rot_log(T_goal[:3, :3] @ T_cur[:3, :3].T)])


def _load_spheres(path: str = _SPHERE_FILE):
    with open(path) as f:
        data = json.load(f)
    centers, radii, ids = [], [], []
    for name in LINK_NAMES:
        if name not in data["links"]:
            continue
        c = np.asarray(data["links"][name]["centers"], dtype=float).reshape(-1, 3)
        r = np.asarray(data["links"][name]["radii"], dtype=float).reshape(-1)
        centers.append(c); radii.append(r); ids.append(np.full(len(r), LINK_INDEX[name]))
    return np.concatenate(centers), np.concatenate(radii), np.concatenate(ids).astype(np.int64), data


class Panda:
    """Kinematic model of the Franka Panda + hand (Isaac Lab body frames, world = panda_link0).

    qd_max: 0.8 x the official velocity limits [rad/s]; q_lower, q_upper [rad].
    sphere_groups: sphere indices of 'arm' (links 1-7), 'hand' and 'fingers'.
    """

    joint_names = [f"panda_joint{i}" for i in range(1, 8)]
    link_names = LINK_NAMES

    def __init__(self, sphere_file: str | None = _SPHERE_FILE):
        self.q_lower = Q_LOWER.copy()
        self.q_upper = Q_UPPER.copy()
        self.qd_max = 0.8 * QD_MAX_OFFICIAL
        self.q_home = Q_HOME.copy()
        self.tcp_offset = TCP_OFFSET
        self.max_width = MAX_WIDTH

        # A_i = Trans(p_i) Rx(alpha_i) (4x4); the joint transform is A_i Rz(q_i)
        A = np.tile(np.eye(4), (7, 1, 1))
        for i in range(7):
            A[i, :3, :3] = _rx(_JOINT_RX[i])
            A[i, :3, 3] = _JOINT_XYZ[i]
        self._A = A
        T7h = np.eye(4)
        T7h[:3, :3] = rotz(-np.pi / 4)
        T7h[2, 3] = FLANGE_Z
        self._T_7_hand = T7h
        self._T_hand_tcp = np.eye(4)
        self._T_hand_tcp[2, 3] = TCP_OFFSET
        # number of arm joints that move each link (link0: 0 ... link7/hand/fingers: 7)
        self._n_joints_of_link = np.array([0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7])

        if sphere_file is not None and os.path.exists(sphere_file):
            c, r, ids, data = _load_spheres(sphere_file)
            self._sph_local, self.sphere_radii, self.sphere_link_ids = c, r, ids
            self.sphere_meta = {k: v for k, v in data.items() if k not in ("links", "hulls")}
            self._sph_mask = np.arange(7)[None, :] < self._n_joints_of_link[ids][:, None]  # (S,7)
            self.sphere_groups = {
                "arm": np.flatnonzero((ids >= 1) & (ids <= 7)),
                "hand": np.flatnonzero(ids == HAND_ID),
                "fingers": np.flatnonzero((ids == LEFT_ID) | (ids == RIGHT_ID)),
            }
            self.collision_hulls = {k: np.asarray(v) for k, v in data.get("hulls", {}).items()}
        else:  # kinematics still usable without the sphere model
            self._sph_local = np.zeros((0, 3)); self.sphere_radii = np.zeros(0)
            self.sphere_link_ids = np.zeros(0, dtype=np.int64); self._sph_mask = np.zeros((0, 7), bool)
            self.sphere_groups = {"arm": np.zeros(0, int), "hand": np.zeros(0, int), "fingers": np.zeros(0, int)}
            self.collision_hulls = {}
        # spheres() returns these without copying; read-only stops callers editing the shared model
        for a in (self._sph_local, self.sphere_radii, self.sphere_link_ids, self._sph_mask,
                  *self.sphere_groups.values()):
            a.flags.writeable = False

    @property
    def n_spheres(self) -> int:
        return len(self.sphere_radii)

    def link_transforms(self, q, width=MAX_WIDTH) -> np.ndarray:
        """World poses of all 11 bodies in LINK_NAMES order: q (...,7), width -> T (...,11,4,4).
        width [m] is clipped to the finger range [0, MAX_WIDTH]."""
        q = np.asarray(q, dtype=float)
        if q.ndim == 1:
            # plain-python clip: np.clip on a scalar is slower than the whole FK; NaN passes
            return self._link_transforms_single(q, min(max(float(width), 0.0), MAX_WIDTH))
        width = np.clip(width, 0.0, MAX_WIDTH)
        batch = q.shape[:-1]
        c, s = np.cos(q)[..., :, None], np.sin(q)[..., :, None]      # (...,7,1)
        A = self._A
        M = np.empty(batch + (7, 4, 4))
        M[..., 0] = c * A[:, :, 0] + s * A[:, :, 1]
        M[..., 1] = -s * A[:, :, 0] + c * A[:, :, 1]
        M[..., 2:] = A[:, :, 2:]
        T = np.empty(batch + (11, 4, 4))
        T[..., 0, :, :] = np.eye(4)
        T[..., 1, :, :] = M[..., 0, :, :]
        for j in range(1, 7):
            T[..., j + 1, :, :] = T[..., j, :, :] @ M[..., j, :, :]
        T[..., HAND_ID, :, :] = T[..., 7, :, :] @ self._T_7_hand
        # fingers: same orientation as the hand, origin at hand + R_hand (0, +-w/2, 0.0584)
        Th = T[..., HAND_ID, :, :]
        half = 0.5 * np.asarray(width, dtype=float)[..., None]
        base = Th[..., :3, 3] + FINGER_Z * Th[..., :3, 2]
        for fid, sign in ((LEFT_ID, 1.0), (RIGHT_ID, -1.0)):
            T[..., fid, :, :] = Th
            T[..., fid, :3, 3] = base + sign * half * Th[..., :3, 1]
        return T

    def _link_transforms_single(self, q, width):
        """Unbatched fast path of :meth:`link_transforms` (same math, fewer numpy calls)."""
        A = self._A
        c, s = np.cos(q)[:, None], np.sin(q)[:, None]
        M = A.copy()
        M[:, :, 0] = c * A[:, :, 0] + s * A[:, :, 1]
        M[:, :, 1] = c * A[:, :, 1] - s * A[:, :, 0]
        T = np.empty((11, 4, 4))
        T[0] = np.eye(4)
        T[1] = M[0]
        for j in range(1, 7):
            np.matmul(T[j], M[j], out=T[j + 1])
        np.matmul(T[7], self._T_7_hand, out=T[HAND_ID])
        T[LEFT_ID] = T[HAND_ID]
        T[RIGHT_ID] = T[HAND_ID]
        base = T[HAND_ID, :3, 3] + FINGER_Z * T[HAND_ID, :3, 2]
        h = (0.5 * width) * T[HAND_ID, :3, 1]
        T[LEFT_ID, :3, 3] = base + h
        T[RIGHT_ID, :3, 3] = base - h
        return T

    def fk_all(self, q, width=MAX_WIDTH) -> dict[str, np.ndarray]:
        """dict body name -> (4,4) world pose; batched q (...,7) gives (...,4,4) per body."""
        T = self.link_transforms(q, width)
        return {n: T[..., i, :, :] for i, n in enumerate(LINK_NAMES)}

    def tcp(self, q) -> np.ndarray:
        """4x4 TCP pose in world: T_hand · Trans(0, 0, 0.1034).  q may be batched (...,7)."""
        T = self.link_transforms(q)[..., HAND_ID, :, :].copy()
        T[..., :3, 3] += TCP_OFFSET * T[..., :3, 2]
        return T

    fk_tcp = tcp

    def _axes_origins(self, T):
        """World joint axes z_j and origins o_j of joints 1..7: T (...,11,4,4) -> (...,7,3) each."""
        return T[..., 1:8, :3, 2], T[..., 1:8, :3, 3]

    def jacobian_point(self, q, link: str | int, p_world, width=MAX_WIDTH) -> np.ndarray:
        """(3,7) linear Jacobian of a point fixed to ``link`` at p_world: J[:, j] = z_j x (p - o_j).
        p_world (N,3) gives (N,3,7); finger links use the hand's columns."""
        lid = LINK_INDEX[link] if isinstance(link, str) else int(link)
        T = self.link_transforms(q, width)
        z, o = self._axes_origins(T)
        p = np.asarray(p_world, dtype=float)
        J = np.cross(z, p[..., None, :] - o)                                   # (...,7,3)
        J[..., self._n_joints_of_link[lid]:, :] = 0.0
        return np.swapaxes(J, -1, -2)                                          # (...,3,7)

    def jacobian_tcp(self, q) -> np.ndarray:
        """(6,7) geometric Jacobian at the TCP, world frame: rows [v; w], J_w[:, j] = z_j."""
        return self._tcp_and_jac(q)[1]

    def _tcp_and_jac(self, q):
        T = self.link_transforms(np.asarray(q, dtype=float).reshape(7))
        Th = T[HAND_ID]
        Ttcp = Th.copy()
        Ttcp[:3, 3] += TCP_OFFSET * Th[:3, 2]
        zt = T[1:8, :3, 2].T                                   # (3,7) joint axes
        dt = Ttcp[:3, 3:4] - T[1:8, :3, 3].T                   # (3,7) TCP - joint origins
        J = np.empty((6, 7))
        J[0] = zt[1] * dt[2] - zt[2] * dt[1]                   # z_j x (p - o_j), row-wise
        J[1] = zt[2] * dt[0] - zt[0] * dt[2]
        J[2] = zt[0] * dt[1] - zt[1] * dt[0]
        J[3:] = zt
        return Ttcp, J

    def spheres(self, q, width=MAX_WIDTH):
        """Collision spheres at (q, width): centres (S,3) world, radii (S,), jac (S,3,7), link_ids.

        jac = d centre / dq over the arm joints (fingers rigid with the hand). Batched q (...,7)
        adds leading axes to centres and jac. radii and link_ids are the model's read-only arrays.
        """
        T = self.link_transforms(q, width)
        ids = self.sphere_link_ids
        Ts = T[..., ids, :, :]              # gather first: T[..., ids, :3, 3] would move the ids axis first
        centers = np.einsum("...sij,sj->...si", Ts[..., :3, :3], self._sph_local) + Ts[..., :3, 3]
        z, o = self._axes_origins(T)                                           # (...,7,3)
        J = np.cross(z[..., None, :, :], centers[..., :, None, :] - o[..., None, :, :])   # (...,S,7,3)
        J *= self._sph_mask[:, :, None]
        return centers, self.sphere_radii, np.swapaxes(J, -1, -2), ids

    def sphere_centers(self, q, width=MAX_WIDTH) -> np.ndarray:
        """Sphere centres only, q (...,7) and width (...) -> (...,S,3); cheap for many poses."""
        Ts = self.link_transforms(q, width)[..., self.sphere_link_ids, :, :]   # (..., S, 4, 4)
        return np.einsum("...sij,sj->...si", Ts[..., :3, :3], self._sph_local) + Ts[..., :3, 3]

    def clamp(self, q) -> np.ndarray:
        return np.clip(q, self.q_lower, self.q_upper)

    def random_q(self, rng: np.random.Generator, n: int | None = None, margin: float = 0.0) -> np.ndarray:
        """Uniform random configuration(s) within limits shrunk by ``margin`` rad."""
        lo, hi = self.q_lower + margin, self.q_upper - margin
        size = (7,) if n is None else (n, 7)
        return lo + (hi - lo) * rng.random(size)

    def ik(self, T_target, q_seed, max_iters=200, pos_tol=2e-3, rot_tol=2e-2, *, rot_weight=0.5,
           damping=1e-4, stall_iters=25, return_info=False):
        """TCP IK by Levenberg-Marquardt (Sugihara 2011 damping) within joint limits -> (q, ok).

        Joints on a limit whose step points outward are frozen and the step re-solved. Runs to
        0.05 x tolerance (cheap with quadratic convergence) or stall_iters without 2 % progress.
        ok: best iterate within pos_tol [m] and rot_tol [rad]; a non-finite target gives ok=False.
        """
        T_target = np.asarray(T_target, dtype=float)
        q = np.array(q_seed, dtype=float).reshape(7)
        bad = ~np.isfinite(q)
        if bad.any():
            q[bad] = self.q_home[bad]
        q = self.clamp(q)
        if not np.isfinite(T_target).all():
            return (q, False, dict(iters=0, pos_err=np.inf, rot_err=np.inf)) if return_info else (q, False)
        wv = np.array([1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight])
        lo, hi = self.q_lower + 1e-9, self.q_upper - 1e-9
        best_q, best_E, best_err = q.copy(), np.inf, (np.inf, np.inf)
        stall, it = 0, 0
        for it in range(max_iters):
            Tc, J = self._tcp_and_jac(q)
            e = pose_error(Tc, T_target)
            ep, er = np.linalg.norm(e[:3]), np.linalg.norm(e[3:])
            E = 0.5 * float(e @ (wv * e))
            if E < best_E:
                stall = 0 if E < 0.98 * best_E else stall + 1
                best_q, best_E, best_err = q.copy(), E, (ep, er)
            else:
                stall += 1
            if (ep < 0.05 * pos_tol and er < 0.05 * rot_tol) or stall > stall_iters:
                break
            lam = E + damping
            JW = J.T * wv                                          # J^T W  (7,6)
            dq = np.linalg.solve(JW @ J + lam * np.eye(7), JW @ e)
            blocked = ((q <= lo) & (dq < 0)) | ((q >= hi) & (dq > 0))
            if blocked.any():                                      # active-set re-solve
                free = ~blocked
                Jf, JWf = J[:, free], JW[free]
                dq = np.zeros(7)
                dq[free] = np.linalg.solve(JWf @ Jf + lam * np.eye(int(free.sum())), JWf @ e)
            q = self.clamp(q + dq)
        ok = bool(best_err[0] < pos_tol and best_err[1] < rot_tol)
        if return_info:
            return best_q, ok, dict(iters=it + 1, pos_err=float(best_err[0]), rot_err=float(best_err[1]))
        return best_q, ok

    def default_ik_seeds(self) -> np.ndarray:
        """(7,7) IK seeds: q_home, base turned +-1.6 / +-2.6 rad, stretched-low and folded-high."""
        h = self.q_home
        S = np.tile(h, (7, 1))
        S[1, 0], S[2, 0] = 1.6, -1.6
        S[3, [1, 3, 5, 6]] = [0.6, -1.2, 1.5, -0.8]
        S[4, [0, 1]] = [2.6, 0.3]
        S[5, [0, 1]] = [-2.6, 0.3]
        S[6, [1, 3, 5]] = [-1.0, -2.8, 3.2]
        return S

    def ik_multi(self, T_target, seeds=None, max_iters=200, pos_tol=2e-3, rot_tol=2e-2, *, q_ref=None,
                 return_first=True, flip_z=False, rng=None, n_random=0, **kw):
        """IK from several seeds; returns (q, ok), or the lowest-error attempt with ok=False.

        seeds: (N,7); None -> default_ik_seeds() with q_ref first. n_random adds uniform seeds.
        return_first: take the first success (seed order = preference), else the one nearest q_ref.
        flip_z: also try the target turned by pi about its z axis (same grasp, fingers swapped).
        """
        if seeds is not None:
            seeds = list(np.asarray(seeds, dtype=float).reshape(-1, 7))
        if seeds is None or (not seeds and n_random <= 0):
            seeds = list(self.default_ik_seeds())
            if q_ref is not None:      # q_ref first: otherwise return_first would ignore it entirely
                seeds.insert(0, np.asarray(q_ref, dtype=float).reshape(7))
        if n_random > 0:
            rng = rng if rng is not None else np.random.default_rng(0)
            seeds += list(self.random_q(rng, n_random, margin=0.1))
        targets = [np.asarray(T_target, dtype=float)]
        if flip_z:
            Tf = targets[0].copy()
            Tf[:3, :2] = -Tf[:3, :2]                                # T · Rz(pi)
            targets.append(Tf)
        q_ref = seeds[0] if q_ref is None else np.asarray(q_ref, dtype=float)
        best, best_fail = None, None
        for s in seeds:
            for Tt in targets:
                q, ok, info = self.ik(Tt, s, max_iters, pos_tol, rot_tol, return_info=True, **kw)
                if ok:
                    if return_first:
                        return q, True
                    d = float(np.linalg.norm(q - q_ref))
                    if best is None or d < best[0]:
                        best = (d, q)
                else:
                    err = info["pos_err"] + 0.1 * info["rot_err"]
                    if best_fail is None or err < best_fail[0]:
                        best_fail = (err, q)
        if best is not None:
            return best[1], True
        return best_fail[1], False

    def manipulability(self, q) -> float:
        """Yoshikawa manipulability sqrt(det(J J^T)) of the TCP Jacobian."""
        J = self.jacobian_tcp(q)
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def in_limits(self, q, tol=1e-9) -> bool:
        q = np.asarray(q)
        return bool(np.all(q >= self.q_lower - tol) and np.all(q <= self.q_upper + tol))
