"""Nominal TCP controller with no obstacle avoidance: damped least-squares pose tracking plus a
null-space posture task. The safety layers (APF, CBF) then modify its output."""
from __future__ import annotations

import numpy as np

from safegrasp.kinematics.panda import pose_error as _pose_error
from safegrasp.kinematics.panda import rot_log  # noqa: F401  (re-export for convenience)

__all__ = ["pose_error", "pose_error_norms", "pose_reached", "saturate", "dls_pinv",
           "nominal_qd", "scale_to_limits"]


def pose_error(T_cur: np.ndarray, T_goal: np.ndarray) -> np.ndarray:
    """(6,) world-frame error ``[p_goal - p_cur ; rotvec(R_goal R_cur^T)]``, robust near pi."""
    return _pose_error(np.asarray(T_cur, dtype=float), np.asarray(T_goal, dtype=float))


def pose_error_norms(T_cur: np.ndarray, T_goal: np.ndarray) -> tuple[float, float]:
    """(position error [m], rotation error [rad]) between two poses."""
    e = pose_error(T_cur, T_goal)
    return float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))


def pose_reached(T_cur: np.ndarray, T_goal: np.ndarray, pos_tol: float = 0.005,
                 rot_tol: float = 0.05) -> bool:
    """Convergence check: ``|e_p| < pos_tol`` and ``|e_r| < rot_tol``."""
    ep, er = pose_error_norms(T_cur, T_goal)
    return ep < pos_tol and er < rot_tol


def saturate(x: np.ndarray, max_norm: float) -> np.ndarray:
    """Scale ``x`` down so that ``|x| <= max_norm`` (direction preserved)."""
    n = float(np.linalg.norm(x))
    return x * (max_norm / n) if n > max_norm else x


def scale_to_limits(qd: np.ndarray, qd_max: np.ndarray) -> np.ndarray:
    """Uniformly scale ``qd`` so that every ``|qd_j| <= qd_max_j`` (keeps the motion direction)."""
    s = float(np.max(np.abs(qd) / qd_max))
    return qd / s if s > 1.0 else qd


def dls_pinv(J: np.ndarray, damping: float = 0.05) -> np.ndarray:
    """Damped least-squares inverse ``J^T (J J^T + lambda^2 I)^{-1}``, (n,m) for J (m,n).
    Keeps joint speeds bounded near singularities."""
    m = J.shape[0]
    return np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(m), J).T


def nominal_qd(panda, q, T_goal, kp_pos: float = 2.0, kp_rot: float = 2.0, v_max: float = 0.25,
               w_max: float = 1.0, q_posture=None, k_null: float = 0.5, damping: float = 0.05,
               *, return_info: bool = False):
    """Nominal joint velocity (7,) that drives the TCP toward T_goal (4,4) in the world frame.

    kp_pos, kp_rot, k_null [1/s]; v_max [m/s] and w_max [rad/s] cap the TCP twist.
    q_posture: (7,) null-space posture target (default panda.q_home); damping: DLS lambda.
    Returns zero (info['valid'] = False) if q, T_goal or q_posture are non-finite.
    """
    q = np.asarray(q, dtype=float).reshape(7)
    T_goal = np.asarray(T_goal, dtype=float)
    q_post = panda.q_home if q_posture is None else np.asarray(q_posture, dtype=float)
    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(T_goal))
            and (k_null <= 0.0 or np.all(np.isfinite(q_post)))):
        qd = np.zeros(7)
        if return_info:
            nan6 = np.full(6, np.nan)
            return qd, dict(err=nan6, xd=nan6, qd_null=np.zeros(7), pos_err=float("nan"),
                            rot_err=float("nan"), valid=False)
        return qd
    if hasattr(panda, "_tcp_and_jac"):                     # one FK pass for pose + Jacobian
        T_cur, J = panda._tcp_and_jac(q)                   # (4,4), (6,7)
    else:
        T_cur, J = panda.tcp(q), panda.jacobian_tcp(q)
    e = pose_error(T_cur, T_goal)
    xd = np.concatenate([saturate(kp_pos * e[:3], v_max), saturate(kp_rot * e[3:], w_max)])
    qd_task = dls_pinv(J, damping) @ xd

    # posture task in the null space of J, so it adds no TCP motion
    qd_null = np.zeros(7)
    if k_null > 0.0:
        _, sv, Vt = np.linalg.svd(J)                       # Vt (7,7)
        rank = int(np.sum(sv > 1e-6 * max(sv[0], 1e-12)))
        Vn = Vt[rank:]                                     # null-space basis (7-rank, 7)
        qd_null = Vn.T @ (Vn @ (k_null * (q_post - q)))
    qd = scale_to_limits(qd_task + qd_null, panda.qd_max)
    if return_info:
        return qd, dict(err=e, xd=xd, qd_null=qd_null, pos_err=float(np.linalg.norm(e[:3])),
                        rot_err=float(np.linalg.norm(e[3:])), valid=True)
    return qd
