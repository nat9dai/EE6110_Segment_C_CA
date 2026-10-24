"""Run one grasp trial: perceive, propose, execute through a safety layer at 60 Hz, then evaluate.

The pipeline sees only noisy depth, masks and joint state; ground truth is read only by :class:`Monitor`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from safegrasp.control.cbf import Barrier, CBFFilter
from safegrasp.control.apf import APFParams, apf_qd
from safegrasp.control.nominal import nominal_qd, pose_error
from safegrasp.grasping.antipodal import AntipodalConfig, propose_antipodal
from safegrasp.grasping.topdown import TopDownConfig, propose_topdown
from safegrasp.grasping.types import Grasp, pregrasp_pose, to_dict
from safegrasp.kinematics.panda import Panda
from safegrasp.perception.pointcloud import add_depth_noise, backproject_views, estimate_normals
from safegrasp.perception.shadow import shadow_mask
from safegrasp.perception.types import View
from safegrasp.perception.voxel import (DistanceField, VoxelGrid, build_occupancy, label_counts,
                                        obstacle_mask, unknown_mask_for_target)
from safegrasp.pipeline.methods import Method
from safegrasp.pipeline.recovery import RecoveryState, approach_resume_time

WS_ROI = (np.array([0.22, -0.38, 0.0]), np.array([0.88, 0.38, 0.32]))


@dataclass
class TrialCfg:
    ctrl_dt: float = 1.0 / 60.0
    pregrasp_dist: float = 0.12
    approach_speed: float = 0.06        # [m/s] along the approach axis
    lift_height: float = 0.15
    lift_speed: float = 0.10
    t_transit: float = 6.0
    t_approach_extra: float = 2.5       # [s] beyond the nominal approach time
    t_close: float = 0.7
    t_lift_extra: float = 2.0
    t_hold: float = 1.0
    pos_tol: float = 0.008
    grasp_tol: float = 0.003            # [m]
    commit_resid: float = 0.015         # [m] a blocked approach this close still closes
    rot_tol: float = 0.06
    stall_time: float = 1.5             # [s] no progress this long = deadlock
    max_recoveries: int = 2
    q_lead_max: float = 0.08            # anti-windup: |q_cmd - q| per joint [rad]
    noise_sigma0: float = 0.001
    noise_sigma2: float = 0.002
    noise_dropout: float = 0.01
    contact_thresh: float = 1.0         # [N]
    disturb_pos: float = 0.02           # [m]
    disturb_rot: float = 15.0           # [deg]
    success_lift: float = 0.08          # [m]
    max_attempts: int = 3
    t_retreat: float = 3.0
    trace: bool = False
    film_every: int = 0                 # >0: RGB frame every N control steps
    film_cam: str = "main"


@dataclass
class Percept:
    views: list
    grid: VoxelGrid
    labels: np.ndarray
    inst: np.ndarray
    target_pts: np.ndarray
    target_normals: np.ndarray
    obstacle_pts: np.ndarray
    f_obs: DistanceField
    f_unk: DistanceField
    stats: dict


def perceive(views: list[View], target: int, panda: Panda, q_capture, method: Method, rng, cfg: TrialCfg) -> Percept:
    t0 = time.perf_counter()
    noisy = []
    for v in views:
        d = v.depth
        if method.noise > 0:
            d = add_depth_noise(d, rng, sigma0=cfg.noise_sigma0 * method.noise, sigma2=cfg.noise_sigma2 * method.noise,
                                dropout=cfg.noise_dropout * method.noise, edge_dropout=True)
        noisy.append(View(depth=d, K=v.K, T_wc=v.T_wc, seg=v.seg, name=v.name))
    pts, ids, vps = backproject_views(noisy, stride=1)
    tmask = ids == target
    tpts, tvp = pts[tmask], vps[tmask]
    if len(tpts) > 6000:
        sel = rng.choice(len(tpts), 6000, replace=False)
        tpts, tvp = tpts[sel], tvp[sel]
    tnrm = estimate_normals(tpts, k=16, viewpoint=tvp) if len(tpts) >= 16 else np.zeros_like(tpts)
    omask = (ids != target) & (ids >= 0)
    opts = pts[omask]
    if len(opts) > 30000:
        opts = opts[rng.choice(len(opts), 30000, replace=False)]
    grid = VoxelGrid()
    c, r, _, _ = panda.spheres(q_capture, 0.08)
    labels, inst = build_occupancy(noisy, grid, robot_spheres=(c, r))
    f_obs = DistanceField.from_mask(obstacle_mask(labels, inst, target), grid)
    # barrier set: shadowed unknown voxels, minus the target's own shadow and hull
    umask = unknown_mask_for_target(labels, inst, target, grid, carve_radius=method.carve_radius, roi=WS_ROI)
    umask &= shadow_mask(noisy, grid)
    umask &= ~target_hull_mask(tpts, grid)
    f_unk = DistanceField.from_mask(umask, grid)
    stats = dict(n_target_pts=int(tmask.sum()), n_obstacle_pts=int(omask.sum()), **label_counts(labels),
                 n_unk_barrier=int(umask.sum()), perceive_s=time.perf_counter() - t0)
    return Percept(noisy, grid, labels, inst, tpts, tnrm, opts, f_obs, f_unk, stats)


def target_hull_mask(tpts, grid: VoxelGrid, dilate: int = 1) -> np.ndarray:
    """Return voxels inside the hull of the target points and their table projection (likely target)."""
    from scipy.ndimage import binary_dilation
    from scipy.spatial import Delaunay
    m = np.zeros(tuple(grid.shape), dtype=bool)
    if len(tpts) < 8:
        return m
    P = tpts[:: max(1, len(tpts) // 800)]
    base = P.copy()
    base[:, 2] = 0.0
    H = np.concatenate([P, base])
    lo, hi = H.min(0) - 0.02, H.max(0) + 0.02
    C = grid.centers(flat=True).reshape(-1, 3)
    sel = np.all((C >= lo) & (C <= hi), axis=1)
    try:
        inside = Delaunay(H).find_simplex(C[sel]) >= 0
    except Exception:  # degenerate (flat) point set
        return m
    flat = np.zeros(len(C), dtype=bool)
    flat[np.flatnonzero(sel)[inside]] = True
    m = flat.reshape(tuple(grid.shape))
    return binary_dilation(m, iterations=dilate) if dilate else m


def _ik_pair(panda: Panda, g: Grasp, q_seed, pregrasp_dist):
    """Solve IK for the pre-grasp, then for the grasp seeded from it; returns (q_pre, q_g) or None."""
    T_pre = pregrasp_pose(g, pregrasp_dist)
    q_pre, ok1 = panda.ik_multi(T_pre, q_ref=q_seed)
    if not ok1:
        return None
    q_g, ok2 = panda.ik(g.T, q_pre)
    if not ok2:
        q_g, ok2 = panda.ik_multi(g.T, q_ref=q_pre)
    if not ok2 or np.max(np.abs(q_g - q_pre)) > 1.2:
        return None
    return q_pre, q_g


_GEOM = None
# favour near-vertical approaches: tilted grasps slipped 4x more often in development runs
ANTIPODAL_CFG = AntipodalConfig(w_antipodal=0.4, w_center=0.15, w_count=0.1, w_vertical=0.35)


def pad_contact_filter(grasps, target_pts, min_pts=20, min_span=0.020, min_antipodal=0.45,
                       relaxed_span=0.012, depth_bonus=0.3):
    """Keep grasps whose finger pads hold enough observed target points, best score first.

    Strict tier needs ``min_span`` and ``min_antipodal``, else ``relaxed_span``; deeper grasps gain up to
    ``depth_bonus`` (grasps > 11 mm deep succeeded 82 % vs ~22 % in development runs)."""
    global _GEOM
    if _GEOM is None:
        from safegrasp.grasping.gripper import GripperGeometry
        _GEOM = GripperGeometry.load() if hasattr(GripperGeometry, "load") else GripperGeometry()
    gx = 1.5 * float(np.max(np.abs(_GEOM.finger_x)))
    z0, z1 = float(_GEOM.pad_z[0]), float(_GEOM.pad_z[1])
    strict, relaxed = [], []
    for g in grasps:
        R, t = g.T[:3, :3], g.T[:3, 3]
        pl = (target_pts - t) @ R
        m = (np.abs(pl[:, 0]) <= gx) & (np.abs(pl[:, 1]) < 0.5 * g.width) & (pl[:, 2] >= z0) & (pl[:, 2] <= z1)
        span = float(np.ptp(pl[m, 2])) if m.sum() >= 2 else 0.0
        g.info["pad_pts"] = int(m.sum())
        g.info["pad_span"] = span
        if m.sum() < min_pts or span < relaxed_span:
            continue
        g.score = float(g.score) + depth_bonus * float(np.clip(g.info.get("depth", 0.0) / 0.02, 0.0, 1.0))
        if span >= min_span and g.info.get("antipodal", 1.0) >= min_antipodal:
            strict.append(g)
        else:
            relaxed.append(g)
    keep = strict if strict else relaxed
    return sorted(keep, key=lambda g: -g.score)


def propose(method: Method, per: Percept, panda: Panda, q_home, rng, cfg: TrialCfg):
    """Return (candidates [(Grasp, q_pre, q_grasp)] best first, at most cfg.max_attempts; info)."""
    t0 = time.perf_counter()
    info = {}
    if len(per.target_pts) < 20:
        return [], {"fail": "target_not_seen", "propose_s": 0.0}
    if method.proposer == "topdown":
        grasps = propose_topdown(per.target_pts, rng, TopDownConfig(), obstacle_pts=per.obstacle_pts)
    else:
        grasps = propose_antipodal(per.target_pts, per.target_normals, per.obstacle_pts, rng, ANTIPODAL_CFG)
    info["n_candidates_raw"] = len(grasps)
    grasps = pad_contact_filter(grasps, per.target_pts)
    info["n_candidates"] = len(grasps)
    cands = []
    if method.proposer == "safe" and grasps:
        from safegrasp.grasping.safety_rank import SafetyRankConfig, rank_safe
        # penalise blocked or shadow-entering corridors, otherwise keep the proposer's order
        SAFETY_RANK_CFG = SafetyRankConfig(pregrasp_dist=cfg.pregrasp_dist, finger_grace=0.0,
                                           w_clear=0.0, w_unk=0.0, w_joint=0.02)
        fields = {"obs": (per.f_obs, method.margin_obs), "unk": (per.f_unk, method.margin_unk)}
        ranked = rank_safe(grasps, panda, fields, q_home, SAFETY_RANK_CFG)
        info["n_ranked"] = len(ranked)
        for g in ranked[: cfg.max_attempts]:
            cands.append((g, np.asarray(g.info["q_pregrasp"]), np.asarray(g.info["q_grasp"])))
    else:
        for g in grasps[:24]:
            sol = _ik_pair(panda, g, q_home, cfg.pregrasp_dist)
            if sol is None:
                # parallel-jaw symmetry: try the grasp rotated 180 deg about its approach axis
                g2 = g.copy()
                g2.T[:3, 0] *= -1
                g2.T[:3, 1] *= -1
                sol = _ik_pair(panda, g2, q_home, cfg.pregrasp_dist)
                g = g2
            if sol is not None:
                cands.append((g, *sol))
                if len(cands) >= cfg.max_attempts:
                    break
    info["propose_s"] = time.perf_counter() - t0
    if not cands:
        info["fail"] = "no_grasp" if not grasps else "ik_fail"
        return [], info
    g = cands[0][0]
    info["grasp"] = {k: v for k, v in to_dict(g).items() if k != "info"}
    info["grasp_info"] = {k: v for k, v in g.info.items() if not isinstance(v, (list, tuple, np.ndarray))}
    return cands, info


class Monitor:
    def __init__(self, scene, target: int, panda: Panda, cfg: TrialCfg, exclude=()):
        self.scene, self.target, self.panda, self.cfg = scene, target, panda, cfg
        self.p0 = scene.object_poses()
        # non-target objects, minus those that drift on their own (null run)
        self.others = [s for s in np.flatnonzero(scene.active) if s != target and s not in set(exclude)]
        pts = scene.object_points_world(self.others, self.p0)
        self.gt_pts = np.concatenate(pts) if pts else np.zeros((0, 3))
        self.gt_owner = np.concatenate([np.full(len(p), s) for p, s in zip(pts, self.others)]) if pts else np.zeros(0, int)
        self.tree = cKDTree(self.gt_pts) if len(self.gt_pts) else None
        self.first_contact = None
        self.max_force = 0.0
        self.contact_steps = 0
        self.contact_objs = set()
        self.min_clear = np.inf          # [m] GT clearance to non-target objects
        self.min_clear_phase = None
        self.target_push = 0.0           # [m]
        self.last_clear = np.nan

    def step(self, q, width, phase, t):
        F = self.scene.contact_matrix()
        if len(self.others):
            Fo = F[:, self.others]
            fmax = float(Fo.max(initial=0.0))
            if fmax > self.cfg.contact_thresh:
                self.contact_steps += 1
                if self.first_contact is None:
                    self.first_contact = (phase, t)
                for j in np.flatnonzero(Fo.max(axis=0) > self.cfg.contact_thresh):
                    self.contact_objs.add(int(self.others[j]))
            self.max_force = max(self.max_force, fmax)
        if self.tree is not None:
            c, r, _, _ = self.panda.spheres(q, width)
            d, _ = self.tree.query(c)
            m = float(np.min(d - r))
            self.last_clear = m
            if m < self.min_clear:
                self.min_clear, self.min_clear_phase = m, phase

    def finish(self, grasp_T=None):
        p1 = self.scene.object_poses()
        disp = np.linalg.norm(p1[:, :3] - self.p0[:, :3], axis=1)
        dq = np.abs(np.sum(p1[:, 3:] * self.p0[:, 3:], axis=1)).clip(0, 1)
        drot = np.degrees(2 * np.arccos(dq))
        oth = np.array(self.others, dtype=int)
        disturbed = [int(s) for s in oth if disp[s] > self.cfg.disturb_pos or drot[s] > self.cfg.disturb_rot]
        return dict(
            collision=self.first_contact is not None,
            first_contact_phase=None if self.first_contact is None else self.first_contact[0],
            max_contact_force=self.max_force,
            contact_time=self.contact_steps * self.scene.cfg.dt * self.scene.cfg.decimation,
            n_contact_objs=len(self.contact_objs),
            n_disturbed=len(disturbed),
            max_disp_other=float(disp[oth].max(initial=0.0)) if len(oth) else 0.0,
            sum_disp_other=float(disp[oth].sum()) if len(oth) else 0.0,
            min_gt_clearance=float(self.min_clear),
            min_gt_clearance_phase=self.min_clear_phase,
            target_push=float(self.target_push),
        )


class SafetyLayer:
    def __init__(self, method: Method, panda: Panda, per: Percept):
        self.m, self.panda, self.per = method, panda, per
        self.cbf = CBFFilter(panda, alpha=method.alpha, alpha_q=5.0)
        self.apf = APFParams(eta=2e-6, max_force=0.3)  # ~0.1 m/s repulsion at 2 cm clearance
        g = panda.sphere_groups
        n = panda.n_spheres
        self.mask_arm = np.zeros(n, bool)
        self.mask_arm[g["arm"]] = True
        self.mask_arm_hand = self.mask_arm.copy()
        self.mask_arm_hand[g["hand"]] = True
        self.mask_all = np.ones(n, bool)
        self.stats = dict(qp_ms=[], intervention=[], h_obs=[], h_unk=[], slack_unk=[], infeasible=0)

    def barriers(self, phase, near_grasp):
        m = self.m
        bs = [Barrier(self.per.f_obs, m.margin_obs, soft=False, name="obs", sphere_mask=self.mask_all),
              Barrier("table", m.margin_table, soft=False, name="table", sphere_mask=self.mask_arm_hand)]
        if m.executor == "oacbf":
            # once the fingers close on the target, only the arm links keep the unknown-space barrier
            unk_mask = self.mask_arm if phase in ("close", "lift", "hold") else self.mask_all
            bs.append(Barrier(self.per.f_unk, m.margin_unk, soft=m.unk_soft, name="unk", sphere_mask=unk_mask))
        return bs

    def __call__(self, q, qd_nom, width, phase, near_grasp, extra):
        ex = self.m.executor
        if ex == "nominal":
            return np.clip(qd_nom, -self.panda.qd_max, self.panda.qd_max)
        bs = self.barriers(phase, near_grasp)
        if ex == "apf":
            return apf_qd(self.panda, q, qd_nom, width, bs, self.apf, extra_spheres=extra)
        qd, info = self.cbf.filter(q, qd_nom, width, bs, extra_spheres=extra)
        s = self.stats
        s["qp_ms"].append(info.get("solve_ms", 0.0))
        s["intervention"].append(info.get("intervention", 0.0))
        hm = info.get("h_min", {})
        s["h_obs"].append(hm.get("obs", np.nan))
        if "unk" in hm:
            s["h_unk"].append(hm["unk"])
        sl = info.get("slack", {})
        if "unk" in sl:
            s["slack_unk"].append(float(np.max(sl["unk"])) if np.size(sl["unk"]) else 0.0)
        if info.get("status", "ok") not in ("ok", "optimal", "solved"):
            s["infeasible"] += 1
        return qd


def _attached_spheres(panda: Panda, q, T_hand_obj_pts, radius=0.012):
    """Return (centres, radii, jac) of spheres on the grasped object (points in hand frame)."""
    T_h = panda.fk_all(q)["panda_hand"]
    c = T_hand_obj_pts @ T_h[:3, :3].T + T_h[:3, 3]
    jac = np.stack([panda.jacobian_point(q, "panda_hand", p) for p in c]) if len(c) else np.zeros((0, 3, 7))
    return c, np.full(len(c), radius), jac


def _downsample(pts, voxel):
    if len(pts) == 0:
        return pts
    key = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return pts[idx]


def execute(scene, method: Method, panda: Panda, per: Percept, cands, target: int, mon: Monitor, cfg: TrialCfg,
            reperceive=None):
    """Try the candidates in order until one closes; a blocked transit or approach moves to the next.

    With ``method.recover != 'none'`` a stall first gets up to ``cfg.max_recoveries`` recoveries."""
    dt = cfg.ctrl_dt
    safety = SafetyLayer(method, panda, per)
    rec = RecoveryState(method.recover, method.executor, cfg.max_recoveries)
    trace = [] if cfg.trace else None
    res = dict(phase_reached="transit", fail=None, attempts=[])
    z_target0 = mon.p0[target, 2]
    st_ = dict(q_cmd=scene.robot_q(), t_all=0.0, attached=None, k=0, T_g=np.eye(4))
    frames = [] if cfg.film_every else None
    res["_frames"] = frames
    phases = ["transit", "approach", "close", "lift", "hold", "retreat"]

    def run(phase, goal_fn, t_max, width, done_fn=None, stall=True):
        """60 Hz loop: nominal -> safety layer -> integrate joint command -> 2 physics steps."""
        t, best, t_best = 0.0, np.inf, 0.0
        while t < t_max:
            q = scene.robot_q()
            qd_nom = nominal_qd(panda, q, goal_fn(t))
            extra = None
            if st_["attached"] is not None and method.executor != "nominal":
                extra = _attached_spheres(panda, q, st_["attached"])
            qd = safety(q, qd_nom, width, phase, False, extra)
            q_cmd = np.clip(st_["q_cmd"] + qd * dt, q - cfg.q_lead_max, q + cfg.q_lead_max)
            st_["q_cmd"] = np.clip(q_cmd, panda.q_lower, panda.q_upper)
            scene.control_step(st_["q_cmd"], width, qd)
            t += dt
            st_["t_all"] += dt
            q_new = scene.robot_q()
            mon.step(q_new, scene.gripper_width(), phase, st_["t_all"])
            T_new = panda.tcp(q_new)
            st_["k"] += 1
            if frames is not None and st_["k"] % cfg.film_every == 0:
                frames.append((st_["t_all"], phase, scene.capture([cfg.film_cam], n_render=2, with_rgb=True)[1][0]))
            if trace is not None:
                trace.append(np.r_[st_["t_all"], phases.index(phase), q_new, T_new[:3, 3],
                                   np.linalg.norm(qd - qd_nom), safety.stats["h_obs"][-1] if safety.stats["h_obs"] else np.nan,
                                   safety.stats["h_unk"][-1] if safety.stats["h_unk"] else np.nan, mon.last_clear,
                                   float(scene.contact_matrix()[:, mon.others].max(initial=0.0)) if len(mon.others) else 0.0])
            if done_fn is not None:
                err = done_fn(T_new)
                if err is True:
                    return "done"
                if stall and err is not None:
                    if err < best - 0.002:
                        best, t_best = err, t
                    elif t - t_best > cfg.stall_time:
                        return "stall"
        return "timeout"

    def reached(T_goal, pos_tol=None):
        pt = cfg.pos_tol if pos_tol is None else pos_tol

        def f(T):
            e = pose_error(T, T_goal)
            if np.linalg.norm(e[:3]) < pt and np.linalg.norm(e[3:]) < cfg.rot_tol:
                return True
            return float(np.linalg.norm(e[:3]) + 0.05 * np.linalg.norm(e[3:]))
        return f

    def recover(phase):
        """Apply the recovery action in place: the safety layer reads safety.per / safety.m every step."""
        info = {}
        if rec.mode == "reobserve":
            new = reperceive()
            info.update(n_unk_barrier_before=int(safety.per.stats["n_unk_barrier"]),
                        n_unk_barrier_after=int(new.stats["n_unk_barrier"]), perceive_s=float(new.stats["perceive_s"]))
            safety.per = new
        elif rec.mode == "relax":
            info["margin_unk_before"] = float(safety.m.margin_unk)
            safety.m = safety.m.with_(margin_unk=0.0)
        return rec.record(phase, st_["t_all"], **info)

    committed = None
    for ai, (g, q_pre, q_g) in enumerate(cands[: cfg.max_attempts]):
        T_pre = pregrasp_pose(g, cfg.pregrasp_dist)
        T_g = g.T.copy()
        width_open = float(np.clip(g.width, 0.02, 0.08))
        st = run("transit", lambda t: T_pre, cfg.t_transit, width_open, reached(T_pre))
        while rec.should_recover(st, "transit"):
            ev = recover("transit")
            st = ev["after"] = run("transit", lambda t: T_pre, cfg.t_transit, width_open, reached(T_pre))
        if st != "done":
            res["attempts"].append(f"transit_{st}")
            continue  # no retreat: the next candidate starts from here
        res["phase_reached"] = "approach"
        L, a = cfg.pregrasp_dist, T_g[:3, 2]
        T_dur = L / cfg.approach_speed

        def ref(t, T_g=T_g, a=a, L=L, T_dur=T_dur):
            T = T_g.copy()
            T[:3, 3] = T_g[:3, 3] - a * L * max(0.0, 1.0 - t / T_dur)
            return T

        def resid_now(T_g=T_g):
            return float(np.linalg.norm(pose_error(panda.tcp(scene.robot_q()), T_g)[:3]))
        st = run("approach", ref, T_dur + cfg.t_approach_extra, width_open, reached(T_g, cfg.grasp_tol))
        # a near-complete approach is closed on anyway, so only recover the others
        while rec.should_recover(st, "approach") and resid_now() > cfg.commit_resid:
            ev = recover("approach")
            t0 = approach_resume_time(panda.tcp(scene.robot_q())[:3, 3], T_g[:3, 3], a, L, T_dur)
            ev["t_resume"] = t0
            st = ev["after"] = run("approach", lambda t, t0=t0: ref(t + t0), T_dur - t0 + cfg.t_approach_extra,
                                   width_open, reached(T_g, cfg.grasp_tol))
        resid = resid_now()
        if st == "done" or resid <= cfg.commit_resid:
            committed = (g, T_g, st, resid)
            res["attempts"].append(f"approach_{st}")
            break
        res["attempts"].append(f"approach_{st}")
        res["approach_residual"] = resid
        if ai + 1 < min(len(cands), cfg.max_attempts):
            run("retreat", lambda t, T_pre=T_pre: T_pre, cfg.t_retreat, width_open, reached(T_pre, 0.02))
    res["n_attempts"] = len(res["attempts"])
    res["recoveries"] = rec.n
    res["recovery_events"] = rec.events
    mon.target_push = float(np.linalg.norm(scene.object_poses()[target, :3] - mon.p0[target, :3]))
    if committed is None:
        res["fail"] = res["attempts"][-1] if res["attempts"] else "no_attempt"
        res["exec_sim_s"] = st_["t_all"]
        return _finish(res, mon, safety, trace, scene, target, z_target0, cfg)
    g, T_g, st, resid = committed
    res["approach_status"] = st
    res["approach_residual"] = resid
    res["grasp_used"] = {k: v for k, v in to_dict(g).items() if k != "info"}
    res["grasp_used_info"] = {k: v for k, v in g.info.items() if not isinstance(v, (list, tuple, np.ndarray))}
    res["phase_reached"] = "close"
    T_hold = T_g if st == "done" else panda.tcp(scene.robot_q())
    run("close", lambda t: T_hold, cfg.t_close, 0.0)
    # carry the target points (hand frame) as robot spheres for lift and hold
    T_h = panda.fk_all(scene.robot_q())["panda_hand"]
    tp = _downsample(safety.per.target_pts, 0.02)  # may be re-observed after recovery
    st_["attached"] = (tp - T_h[:3, 3]) @ T_h[:3, :3]
    res["phase_reached"] = "lift"
    T_l0 = panda.tcp(scene.robot_q())
    T_l1 = T_l0.copy()
    T_l1[2, 3] += cfg.lift_height
    t_l = cfg.lift_height / cfg.lift_speed

    def lref(t):
        T = T_l0.copy()
        T[2, 3] += cfg.lift_height * min(1.0, t / t_l)
        return T
    run("lift", lref, t_l + cfg.t_lift_extra, 0.0, reached(T_l1), stall=False)
    res["phase_reached"] = "hold"
    T_top = panda.tcp(scene.robot_q())
    run("hold", lambda t: T_top, cfg.t_hold, 0.0)
    res["exec_sim_s"] = st_["t_all"]
    return _finish(res, mon, safety, trace, scene, target, z_target0, cfg)


def _finish(res, mon, safety, trace, scene, target, z0, cfg):
    p = scene.object_poses()
    lifted = float(p[target, 2] - z0)
    width = scene.gripper_width()
    tcp_hand = scene.body_pose("panda_hand")[:3]
    res["target_lift"] = lifted
    res["final_width"] = width
    res["success"] = bool(res["fail"] is None and lifted > cfg.success_lift and width > 0.002)
    if res["fail"] is None and not res["success"]:
        res["fail"] = "slip" if lifted < cfg.success_lift else "closed_empty"
    res.update(mon.finish())
    s = safety.stats
    res["qp_ms_mean"] = float(np.mean(s["qp_ms"])) if s["qp_ms"] else 0.0
    res["qp_ms_max"] = float(np.max(s["qp_ms"])) if s["qp_ms"] else 0.0
    res["intervention_mean"] = float(np.mean(s["intervention"])) if s["intervention"] else 0.0
    res["h_obs_min_pred"] = float(np.nanmin(s["h_obs"])) if s["h_obs"] else None
    res["h_unk_min_pred"] = float(np.nanmin(s["h_unk"])) if s["h_unk"] else None
    res["slack_unk_max"] = float(np.max(s["slack_unk"])) if s["slack_unk"] else 0.0
    res["qp_infeasible"] = s["infeasible"]
    if trace is not None:
        res["_trace"] = np.array(trace)
    return res


def run_trial(scene, snapshot, target: int, method: Method, panda: Panda, views_all: dict, noise_seed: int,
              cfg: TrialCfg = TrialCfg(), exclude=()) -> dict:
    """Run one trial; ``views_all`` maps camera name -> View captured once per scene.

    ``noise_seed`` gives methods with the same views identical noise (paired trials)."""
    t0 = time.perf_counter()
    scene.restore(snapshot)
    rng = np.random.default_rng(noise_seed)
    q_home = panda.q_home.copy()
    views = [views_all[n] for n in method.views]
    per = perceive(views, target, panda, q_home, method, rng, cfg)

    def reperceive():
        # same seed, so the main view gets the same noise; images were taken at q_home
        v3 = [views_all[n] for n in ("main", "left", "right")]
        return perceive(v3, target, panda, q_home, method, np.random.default_rng(noise_seed), cfg)
    cands, pinfo = propose(method, per, panda, q_home, np.random.default_rng(noise_seed + 1), cfg)
    out = dict(method=method.name, proposer=method.proposer, executor=method.executor, views="+".join(method.views),
               target=int(target), **{f"perc_{k}": v for k, v in per.stats.items()},
               **{k: v for k, v in pinfo.items() if k not in ("grasp", "grasp_info")})
    out["grasp"] = pinfo.get("grasp")
    out["grasp_info"] = pinfo.get("grasp_info")
    mon = Monitor(scene, target, panda, cfg, exclude)
    if not cands:
        out.update(success=False, phase_reached="plan", collision=False, n_disturbed=0, max_contact_force=0.0,
                   min_gt_clearance=None, target_push=0.0, recoveries=0)
        out["wall_s"] = time.perf_counter() - t0
        return out
    t1 = time.perf_counter()
    res = execute(scene, method, panda, per, cands, target, mon, cfg, reperceive=reperceive)
    res["exec_wall_s"] = time.perf_counter() - t1
    out.update(res)
    out["wall_s"] = time.perf_counter() - t0
    return out
