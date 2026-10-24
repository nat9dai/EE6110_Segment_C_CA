"""Safety-aware grasp ranking: prefer grasps the CBF executor can actually carry out.

Each candidate needs IK, then is scored by whole-robot barrier values along its joint-space corridor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import MAX_WIDTH, Grasp, make_cfg, pregrasp_pose

H_FAR = 1.0          # cap for barrier values (no field / nothing within the field's range) [m]

# necessary reach test: the link7 origin is within the summed link lengths of the shoulder (joint 2)
_SHOULDER = np.array([0.0, 0.0, 0.333])
_FLANGE_TO_TCP = 0.107 + 0.1034
_REACH = float(np.hypot(0.316, 0.0825) + np.hypot(0.384, 0.0825) + 0.088)


@dataclass
class SafetyRankConfig:
    top_n: int = 24                   # candidates (best geometric score first) that are evaluated
    pregrasp_dist: float = 0.12       # pre-grasp = grasp - pregrasp_dist * approach  [m]
    n_transit: int = 10               # joint-space samples q_seed -> q_pre (both ends included)
    n_approach: int = 10              # joint-space samples q_pre -> q_grasp (both ends included)
    approach_iters: int = 2           # DLS steps pulling approach samples onto the straight TCP line (0 = off)
    finger_grace: float = 0.04        # 'unk': hand + fingers exempt when |tcp - tcp_grasp| < this [m]
    table_margin: float | None = 0.01  # table barrier margin (arm + hand spheres); None = off
    h_clip: float = 0.03              # barrier saturation for the score  [m]
    w_clear: float = 0.1              # weight of sat(h_min_obs); only a tie-break
    w_unk: float = 0.1
    w_joint: float = 0.1
    penalty: float = 2.0              # approach corridor blocked by obstacles / table
    unk_penalty: float = 1.0          # corridor enters the occlusion shadow (h_min_unk < 0)
    try_flip: bool = True             # also evaluate T Rz(pi) (parallel-jaw symmetry)
    wrist_offsets: tuple = (-np.pi / 2, np.pi / 2)   # joint-7 offsets for extra IK seeds
    max_joint_jump: float = 1.2       # max_j |q_grasp - q_pre| (continuity of the approach) [rad]
    ik_pos_tol: float = 2e-3
    ik_rot_tol: float = 2e-2
    ik_max_iters: int = 200
    method: str = "safe"


@dataclass
class GraspIK:
    """IK solution of one grasp representative (see :func:`grasp_ik_variants`)."""
    grasp: Grasp                      # the grasp as solved (a flipped copy if ``flipped``)
    q_pre: np.ndarray                 # (7,) pre-grasp configuration
    q_grasp: np.ndarray               # (7,) grasp configuration (continuous with q_pre)
    flipped: bool                     # grasp.T = original T Rz(pi)
    joint_dist: float                 # ||q_pre - q_seed|| + ||q_grasp - q_pre||  [rad]


def flip_about_approach(T: np.ndarray) -> np.ndarray:
    """T Rz(pi): hand x and y negated -- the same physical parallel-jaw grasp."""
    T = np.array(T, dtype=np.float64, copy=True)
    T[..., :3, :2] *= -1.0
    return T


def maybe_reachable(T_tcp, slack: float = 0.01) -> bool:
    """Cheap necessary condition for IK: |p_link7 - p_shoulder| <= reach + slack."""
    T_tcp = np.asarray(T_tcp, dtype=float)
    p7 = T_tcp[:3, 3] - _FLANGE_TO_TCP * T_tcp[:3, 2]
    return bool(np.all(np.isfinite(T_tcp)) and np.linalg.norm(p7 - _SHOULDER) <= _REACH + slack)


def ik_seeds(panda, q_seed, cfg: SafetyRankConfig | None = None) -> np.ndarray:
    """Seed list (N,7), best first: q_seed, q_home, then joint-7 offsets of both (no duplicates).

    Joint 7 takes most of the yaw about the approach, so the offsets start IK in the right basin.
    """
    cfg = make_cfg(SafetyRankConfig, cfg)
    base = [np.asarray(q_seed, dtype=float).reshape(7), np.asarray(panda.q_home, dtype=float)]
    seeds = list(base)
    for q in base:
        for off in cfg.wrist_offsets:
            s = q.copy()
            s[6] = np.clip(s[6] + off, panda.q_lower[6], panda.q_upper[6])
            seeds.append(s)
    out = []
    for s in seeds:
        if not any(np.max(np.abs(s - o)) < 1e-3 for o in out):
            out.append(s)
    return np.stack(out)


def _wrist_flip(panda, q_pre, q_grasp):
    """(q_pre, q_grasp) with q_7 shifted by +-pi (reaches T Rz(pi)), or None if outside joint-7 limits."""
    lo, hi = panda.q_lower[6], panda.q_upper[6]
    for d in ((-np.pi, np.pi) if q_pre[6] > 0 else (np.pi, -np.pi)):
        a, b = q_pre.copy(), q_grasp.copy()
        a[6] += d
        b[6] += d
        if lo <= a[6] <= hi and lo <= b[6] <= hi:
            return a, b
    return None


def ik_pregrasp_grasp(panda, T_grasp, q_seed, cfg: SafetyRankConfig | None = None, seeds=None):
    """IK of the pre-grasp and grasp TCP poses -> (q_pre, q_grasp), or None.

    q_grasp is seeded from q_pre for a continuous approach; None if a joint jumps over max_joint_jump.
    """
    cfg = make_cfg(SafetyRankConfig, cfg)
    kw = dict(max_iters=cfg.ik_max_iters, pos_tol=cfg.ik_pos_tol, rot_tol=cfg.ik_rot_tol)
    seeds = ik_seeds(panda, q_seed, cfg) if seeds is None else np.asarray(seeds, dtype=float).reshape(-1, 7)
    T_grasp = np.asarray(T_grasp, dtype=float)
    T_pre = pregrasp_pose(T_grasp, cfg.pregrasp_dist)
    if not (maybe_reachable(T_grasp) and maybe_reachable(T_pre)):
        return None
    q_pre, ok = panda.ik_multi(T_pre, seeds, **kw)
    if not ok:
        return None
    q_g, ok = panda.ik(T_grasp, q_pre, **kw)
    if not ok:
        q_g, ok = panda.ik_multi(T_grasp, np.vstack([q_pre[None], seeds]), q_ref=q_pre, **kw)
    if not ok or np.max(np.abs(q_g - q_pre)) > cfg.max_joint_jump:
        return None
    return q_pre, q_g


def _joint_dist(q_seed, q_pre, q_g) -> float:
    return float(np.linalg.norm(q_pre - q_seed) + np.linalg.norm(q_g - q_pre))


def grasp_ik_variants(panda, grasp: Grasp, q_seed, cfg: SafetyRankConfig | None = None) -> list[GraspIK]:
    """IK solutions of grasp and, with cfg.try_flip, of T Rz(pi): 0-2 GraspIK entries.

    q_seed is tried first, since a controller starting there reaches the same local solution.
    """
    cfg = make_cfg(SafetyRankConfig, cfg)
    q_seed = np.asarray(q_seed, dtype=float).reshape(7)
    seeds = ik_seeds(panda, q_seed, cfg)

    def with_flip_seed(sol):
        wf = None if sol is None else _wrist_flip(panda, *sol)
        return seeds if wf is None else np.vstack([seeds, wf[0][None]])

    Ts = [np.asarray(grasp.T, dtype=float)]
    if cfg.try_flip:
        Ts.append(flip_about_approach(Ts[0]))
    sols = [ik_pregrasp_grasp(panda, Ts[0], q_seed, cfg, seeds)]
    if cfg.try_flip:
        sols.append(ik_pregrasp_grasp(panda, Ts[1], q_seed, cfg, with_flip_seed(sols[0])))
        if sols[0] is None and sols[1] is not None:
            wf = _wrist_flip(panda, *sols[1])
            if wf is not None:
                sols[0] = ik_pregrasp_grasp(panda, Ts[0], q_seed, cfg, wf[0][None])
    out = []
    for k, (T, sol) in enumerate(zip(Ts, sols)):
        if sol is None:
            continue
        g = grasp.copy()
        g.T = T.copy()
        out.append(GraspIK(g, sol[0], sol[1], bool(k), _joint_dist(q_seed, *sol)))
    return out


def solve_grasp_ik(panda, grasp: Grasp, q_seed, cfg: SafetyRankConfig | None = None) -> GraspIK | None:
    """IK for one grasp without safety checks (same seeding as rank_safe): shorter-joint_dist flip, or None."""
    var = grasp_ik_variants(panda, grasp, q_seed, cfg)
    return min(var, key=lambda v: v.joint_dist) if var else None


def plan_joint_path(q_start, q_goal, n: int = 10) -> np.ndarray:
    """(n,7) straight joint-space path from q_start to q_goal, both ends included."""
    q0 = np.asarray(q_start, dtype=float).reshape(7)
    q1 = np.asarray(q_goal, dtype=float).reshape(7)
    s = np.linspace(0.0, 1.0, max(int(n), 2))[:, None]
    return q0 + s * (q1 - q0)


def corridor(q_seed, q_pre, q_grasp, cfg: SafetyRankConfig | None = None):
    """Transit q_seed -> q_pre then approach q_pre -> q_grasp: samples (M,7) and approach mask (M,)."""
    cfg = make_cfg(SafetyRankConfig, cfg)
    Q = np.vstack([plan_joint_path(q_seed, q_pre, cfg.n_transit), plan_joint_path(q_pre, q_grasp, cfg.n_approach)])
    is_app = np.zeros(len(Q), dtype=bool)
    is_app[max(int(cfg.n_transit), 2):] = True
    return Q, is_app


def straighten_approach(panda, Q, is_approach, T_grasp, pregrasp_dist: float, iters: int = 2):
    """Pull the inner approach samples onto the straight TCP line the executor follows (damped least squares).

    Q: (K,M,7) corridors, is_approach: (M,) mask, T_grasp: (K,4,4). End samples stay unchanged.
    Returns: new (K,M,7) array.
    """
    Q = np.array(Q, dtype=float, copy=True)
    idx = np.flatnonzero(np.asarray(is_approach, dtype=bool))
    if iters <= 0 or len(idx) <= 2:
        return Q
    inner = idx[1:-1]
    K, Mi = Q.shape[0], len(inner)
    T_grasp = np.asarray(T_grasp, dtype=float).reshape(K, 4, 4)
    s = (np.arange(1, len(idx) - 1) / (len(idx) - 1))[None, :, None]          # (1,Mi,1)
    p_g = T_grasp[:, None, :3, 3]
    p_pre = p_g - pregrasp_dist * T_grasp[:, None, :3, 2]
    P_goal = (p_pre + s * (p_g - p_pre)).reshape(-1, 3)                          # (N,3)
    R_goal = np.repeat(T_grasp[:, :3, :3], Mi, axis=0)                           # (N,3,3)
    q = Q[:, inner].reshape(-1, 7)
    hid = panda.link_names.index("panda_hand")
    lam = 1e-4 * np.eye(6)
    for _ in range(int(iters)):
        T = panda.link_transforms(q)
        H = T[:, hid]
        p = H[:, :3, 3] + panda.tcp_offset * H[:, :3, 2]
        z, o = T[:, 1:8, :3, 2], T[:, 1:8, :3, 3]                              # joint axes / origins
        J = np.concatenate([np.cross(z, p[:, None, :] - o), z], axis=2)          # (N,7,6) = J^T
        E = np.einsum("nij,nkj->nik", R_goal, H[:, :3, :3])                      # R* R^T
        er = 0.5 * np.stack([E[:, 2, 1] - E[:, 1, 2], E[:, 0, 2] - E[:, 2, 0], E[:, 1, 0] - E[:, 0, 1]], 1)
        e = np.concatenate([P_goal - p, er], axis=1)                             # (N,6)
        A = np.einsum("nji,njk->nik", J, J) + lam                                # J J^T (N,6,6)
        dq = np.einsum("nji,nj->ni", J.transpose(0, 2, 1), np.linalg.solve(A, e[..., None])[..., 0])
        q = np.clip(q + dq, panda.q_lower, panda.q_upper)
    Q[:, inner] = q.reshape(K, Mi, 7)
    return Q


def _field_margin(entry):
    """fields entry -> (field, margin); a bare field has margin 0, None means nothing there."""
    if isinstance(entry, (tuple, list)):
        return (None, 0.0) if entry[0] is None else (entry[0], float(entry[1]))
    return entry, 0.0


def _field_h(field, margin, centers, radii):
    """h = d(c) - r - margin for centres (...,S,3); non-finite distances -> H_FAR."""
    d, _ = field.query(centers)
    h = np.asarray(d, dtype=float) - radii - float(margin)
    return np.where(np.isfinite(h), np.minimum(h, H_FAR), H_FAR)


def evaluate_corridors(panda, Q, widths, tcp_goal, is_approach, fields: dict, cfg: SafetyRankConfig | None = None):
    """Barrier statistics of K corridors of M joint-space samples, in one batch.

    Q: (K,M,7). widths: (K,). tcp_goal: (K,3) final TCPs. is_approach: (M,) bool.
    fields: {'obs': (field, margin), 'unk': ...}; field.query(pts) returns (signed distance, gradient).
    Returns: dict of (K,) arrays (h_min_* in m, unk_violation_frac, approach_dev).
    """
    cfg = make_cfg(SafetyRankConfig, cfg)
    Q = np.asarray(Q, dtype=float)
    K, M = Q.shape[:2]
    is_app = np.asarray(is_approach, dtype=bool)
    Qf = Q.reshape(-1, 7)
    Wf = np.repeat(np.clip(np.asarray(widths, dtype=float), 0.0, MAX_WIDTH), M)
    C = panda.sphere_centers(Qf, Wf).reshape(K, M, -1, 3)           # (K,M,S,3)
    r = np.asarray(panda.sphere_radii)
    tcp = panda.tcp(Qf)[:, :3, 3].reshape(K, M, 3)
    grp = panda.sphere_groups
    far = np.full((K, M), H_FAR)
    out = {}

    # observed obstacles: every sphere, every sample
    f, m = _field_margin(fields.get("obs"))
    if f is not None:
        h_obs = _field_h(f, m, C, r).min(axis=2)                      # (K,M)
    else:
        h_obs = far
    out["h_min_obs_approach"] = h_obs[:, is_app].min(axis=1)
    out["h_min_obs_transit"] = h_obs[:, ~is_app].min(axis=1) if (~is_app).any() else np.full(K, H_FAR)
    out["h_min_obs"] = h_obs.min(axis=1)

    # unknown space: hand and fingers exempt near the grasp, as they must enter the target's shadow
    f, m = _field_margin(fields.get("unk"))
    if f is not None:
        h = _field_h(f, m, C, r)                                     # (K,M,S)
        near = np.linalg.norm(tcp - np.asarray(tcp_goal, dtype=float)[:, None, :], axis=2) < cfg.finger_grace
        exempt = np.concatenate([grp["hand"], grp["fingers"]])
        he = h[:, :, exempt]
        he[near] = H_FAR
        h[:, :, exempt] = he
        h_unk = h.min(axis=2)                                        # (K,M)
    else:
        h_unk = far
    out["h_min_unk"] = h_unk.min(axis=1)
    out["unk_violation_frac"] = (h_unk < 0).mean(axis=1)
    out["unk_frac"] = out["unk_violation_frac"]

    # table z = 0: arm and hand spheres only, as in the executor
    if cfg.table_margin is not None:
        ah = np.concatenate([grp["arm"], grp["hand"]])
        h_tab = (C[:, :, ah, 2] - r[ah] - cfg.table_margin).min(axis=2)
        h_tab = np.minimum(h_tab, H_FAR)
    else:
        h_tab = far
    out["h_min_table_approach"] = h_tab[:, is_app].min(axis=1)
    out["h_min_table"] = h_tab.min(axis=1)

    # max distance of the approach TCPs from the straight pre-grasp -> grasp segment
    P = tcp[:, is_app]                                               # (K,Ma,3)
    a, b = P[:, :1], P[:, -1:]
    ab = b - a
    t = np.clip(np.sum((P - a) * ab, axis=2) / np.maximum(np.sum(ab * ab, axis=2), 1e-12), 0.0, 1.0)
    out["approach_dev"] = np.linalg.norm(P - (a + t[..., None] * ab), axis=2).max(axis=1)
    return out


def safety_score(geom_score, ev: dict, joint_dist, cfg: SafetyRankConfig | None = None):
    """Geometric score plus clearance terms, minus joint distance and penalties.

    Blocked or shadow-entering corridors fall below every clear one; clearance only breaks ties.
    """
    cfg = make_cfg(SafetyRankConfig, cfg)
    c = cfg.h_clip
    blocked = (ev["h_min_obs_approach"] < 0) | (ev["h_min_table_approach"] < 0)
    return (np.asarray(geom_score, dtype=float)
            + cfg.w_clear * np.clip(ev["h_min_obs"], -c, c) / c
            + cfg.w_unk * np.clip(ev["h_min_unk"], -c, c) / c
            - cfg.w_joint * np.asarray(joint_dist, dtype=float) / np.pi
            - cfg.penalty * blocked
            - cfg.unk_penalty * (np.asarray(ev["h_min_unk"], dtype=float) < 0))


def rank_safe(grasps: list[Grasp], panda, fields: dict, q_seed, cfg: SafetyRankConfig | None = None) -> list[Grasp]:
    """Re-rank the cfg.top_n best grasps by geometric score plus whole-robot barrier values.

    fields: {'obs': (DistanceField, margin), 'unk': ...}; a missing field counts as nothing there.
    q_seed: (7,) current joint configuration.
    Returns: new Grasps with IK, best first, T possibly flipped; info adds the barrier stats and
    q_pregrasp, q_grasp (lists). [] if no candidate has IK.
    """
    cfg = make_cfg(SafetyRankConfig, cfg)
    q_seed = np.asarray(q_seed, dtype=float).reshape(7)
    cand = sorted(grasps, key=lambda g: -g.score)[: max(int(cfg.top_n), 0)]

    var, owner = [], []
    for i, g in enumerate(cand):
        for v in grasp_ik_variants(panda, g, q_seed, cfg):
            var.append(v)
            owner.append(i)
    if not var:
        return []

    # corridors of all variants, evaluated in one batch
    Qs, is_app = zip(*(corridor(q_seed, v.q_pre, v.q_grasp, cfg) for v in var))
    Tg = np.stack([v.grasp.T for v in var])
    Q = straighten_approach(panda, np.stack(Qs), is_app[0], Tg, cfg.pregrasp_dist, cfg.approach_iters)
    ev = evaluate_corridors(panda, Q, [v.grasp.width for v in var], Tg[:, :3, 3], is_app[0], fields, cfg)

    # keep the better flip variant of each candidate
    jd = np.array([v.joint_dist for v in var])
    geom = np.array([cand[i].score for i in owner])
    score = safety_score(geom, ev, jd, cfg)
    best = {}
    for k, i in enumerate(owner):
        if i not in best or (score[k], -jd[k]) > (score[best[i]], -jd[best[i]]):
            best[i] = k

    out = []
    for i, k in best.items():
        v = var[k]
        info = dict(cand[i].info)
        if v.flipped and "y_shift" in info:              # the flip T Rz(pi) negates the closing axis
            info["y_shift"] = -float(info["y_shift"])
        info.update(
            ik_ok=1.0,                                   # Grasp.info holds floats only
            geom_score=float(cand[i].score),
            joint_dist=float(jd[k]),
            flipped=float(v.flipped),
            **{key: float(val[k]) for key, val in ev.items()},
            q_pregrasp=[float(x) for x in v.q_pre],
            q_grasp=[float(x) for x in v.q_grasp],
        )
        out.append(Grasp(T=v.grasp.T.copy(), width=v.grasp.width, score=float(score[k]),
                         method=cfg.method, info=info))
    out.sort(key=lambda g: -g.score)
    return out
