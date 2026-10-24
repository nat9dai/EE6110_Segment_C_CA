"""Unit tests for safegrasp.control."""
from __future__ import annotations

import numpy as np
import pytest

from safegrasp.control.apf import APFParams, apf_qd
from safegrasp.control.cbf import Barrier, CBFFilter, barrier_values
from safegrasp.control.nominal import (dls_pinv, nominal_qd, pose_error, pose_error_norms,
                                       pose_reached)
from safegrasp.kinematics.panda import Panda, pose_error as kin_pose_error, rotz
from safegrasp.perception.voxel import DistanceField, VoxelGrid

DT = 1.0 / 60.0
H_TOL = 2e-3          # allowed CBF violation from discrete steps [m]


class SphereField:
    def __init__(self, c, R):
        self.c, self.R = np.asarray(c, float), float(R)

    def query(self, p):
        v = np.asarray(p, float) - self.c
        n = np.linalg.norm(v, axis=-1)
        return n - self.R, v / np.maximum(n, 1e-12)[..., None]


class PlaneField:
    """Signed distance to the half-space n.(p - p0) < 0."""

    def __init__(self, p0, n):
        self.p0 = np.asarray(p0, float)
        self.n = np.asarray(n, float) / np.linalg.norm(n)

    def query(self, p):
        p = np.asarray(p, float)
        return (p - self.p0) @ self.n, np.broadcast_to(self.n, p.shape).copy()


@pytest.fixture(scope="module")
def panda():
    return Panda()


def goal_pose(panda, xyz, yaw=0.0, tilt_x=0.0):
    """Top-down TCP pose at xyz, with yaw about z and tilt about x."""
    T = panda.tcp(panda.q_home).copy()
    c, s = np.cos(tilt_x), np.sin(tilt_x)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    T[:3, :3] = rotz(yaw) @ Rx @ T[:3, :3]
    T[:3, 3] = xyz
    return T


def rollout(panda, T_goal, steps, safety=None, q0=None, width=0.08, extra_fn=None):
    """Closed-loop rollout; returns q history, qd and filter infos."""
    q = (panda.q_home if q0 is None else q0).copy()
    Q, QD, INFO = [q.copy()], [], []
    for _ in range(steps):
        qn = nominal_qd(panda, q, T_goal)
        extra = extra_fn(q) if extra_fn is not None else None
        if safety is None:
            qd, info = qn, {}
        else:
            qd, info = safety(q, qn, extra)
        q = q + qd * DT
        Q.append(q.copy()); QD.append(qd); INFO.append(info)
    return np.array(Q), np.array(QD), INFO


def min_h(filt, Q, width, barriers, extra_fn=None):
    """Smallest h of each barrier over the trajectory."""
    out = {}
    for q in Q:
        ex = extra_fn(q) if extra_fn is not None else None
        for k, v in filt.evaluate(q, width, barriers, extra_spheres=ex).items():
            out[k] = min(out.get(k, np.inf), v)
    return out


def test_pose_error_translation_and_rotation(panda):
    T0 = goal_pose(panda, [0.5, 0.0, 0.3])
    T1 = T0.copy()
    T1[:3, 3] += [0.01, -0.02, 0.03]
    T1[:3, :3] = rotz(0.3) @ T0[:3, :3]
    e = pose_error(T0, T1)
    assert e.shape == (6,)
    np.testing.assert_allclose(e[:3], [0.01, -0.02, 0.03], atol=1e-12)
    np.testing.assert_allclose(e[3:], [0.0, 0.0, 0.3], atol=1e-9)     # world-frame rotvec
    np.testing.assert_allclose(e, kin_pose_error(T0, T1), atol=1e-12)
    # near pi the rotation vector norm still equals the angle
    T2 = T0.copy()
    T2[:3, :3] = rotz(np.pi - 1e-4) @ T0[:3, :3]
    assert abs(np.linalg.norm(pose_error(T0, T2)[3:]) - (np.pi - 1e-4)) < 1e-6
    ep, er = pose_error_norms(T0, T1)
    assert abs(ep - np.linalg.norm([0.01, -0.02, 0.03])) < 1e-12 and abs(er - 0.3) < 1e-9
    assert pose_reached(T0, T0) and not pose_reached(T0, T1)


def test_dls_pinv_is_damped_least_squares(panda):
    J = panda.jacobian_tcp(panda.q_home)
    lam = 0.05
    xd = np.array([0.1, -0.05, 0.02, 0.1, 0.0, -0.2])
    qd = dls_pinv(J, lam) @ xd
    # normal equations of min |J qd - xd|^2 + lam^2 |qd|^2
    np.testing.assert_allclose((J.T @ J + lam ** 2 * np.eye(7)) @ qd, J.T @ xd, atol=1e-10)


@pytest.mark.parametrize("xyz,yaw,tilt", [([0.55, 0.15, 0.25], np.pi / 4, 0.0),
                                          ([0.40, -0.20, 0.30], -0.5, 0.35),
                                          ([0.65, 0.05, 0.12], 1.2, 0.0)])
def test_nominal_converges_with_saturation(panda, xyz, yaw, tilt):
    Tg = goal_pose(panda, xyz, yaw, tilt)
    q = panda.q_home.copy()
    v_max, w_max = 0.25, 1.0
    for k in range(600):                                   # 10 s
        qd = nominal_qd(panda, q, Tg, v_max=v_max, w_max=w_max)
        assert np.all(np.abs(qd) <= panda.qd_max + 1e-12)
        tw = panda.jacobian_tcp(q) @ qd
        assert np.linalg.norm(tw[:3]) <= v_max + 1e-6     # DLS only shrinks the twist
        assert np.linalg.norm(tw[3:]) <= w_max + 1e-6
        q = q + qd * DT
    ep, er = pose_error_norms(panda.tcp(q), Tg)
    assert ep < 1e-3 and er < 1e-2, (ep, er)


def test_nominal_speed_is_saturated_far_from_goal(panda):
    Tg = goal_pose(panda, [0.45, 0.30, 0.45])              # 30 cm away: kp e = 0.6 > v_max
    qd, info = nominal_qd(panda, panda.q_home, Tg, return_info=True)
    tw = panda.jacobian_tcp(panda.q_home) @ qd
    assert np.linalg.norm(info["xd"][:3]) == pytest.approx(0.25)
    assert 0.23 < np.linalg.norm(tw[:3]) <= 0.25 + 1e-9   # DLS loses a few % only


def test_nullspace_posture_produces_no_tcp_motion(panda):
    q = panda.q_home + np.array([0.3, 0.1, -0.4, 0.1, 0.2, -0.1, 0.3])
    T = panda.tcp(q)                                        # goal = current pose
    qd = nominal_qd(panda, q, T, k_null=0.5)
    J = panda.jacobian_tcp(q)
    assert np.linalg.norm(qd) > 1e-3
    assert np.linalg.norm(J @ qd) < 1e-9
    assert qd @ (panda.q_home - q) > 0
    assert np.allclose(nominal_qd(panda, q, T, k_null=0.0), 0.0, atol=1e-12)


def _cbf(filt, barriers, width=0.08):
    return lambda q, qn, extra: filt.filter(q, qn, width, barriers, extra_spheres=extra)


def _obstacle_on_path_setup(panda):
    """Goal 35 cm along +y, 4 cm ball on the path."""
    Tg = goal_pose(panda, [0.45, 0.35, 0.45])
    ball = SphereField([0.45, 0.17, 0.47], 0.04)
    bars = [Barrier(ball, 0.01, name="obs"), Barrier("table", 0.01, name="table")]
    return Tg, bars


def test_cbf_keeps_safety_on_path_obstacle(panda):
    Tg, bars = _obstacle_on_path_setup(panda)
    filt = CBFFilter(panda)
    Qn, _, _ = rollout(panda, Tg, 480)
    h_nom = min_h(filt, Qn, 0.08, bars)["obs"]
    assert h_nom < -0.03                                    # nominal alone collides
    Q, QD, INFO = rollout(panda, Tg, 480, _cbf(filt, bars))
    h_cbf = min_h(filt, Q, 0.08, bars)
    assert h_cbf["obs"] >= -H_TOL and h_cbf["table"] >= -H_TOL, h_cbf
    assert all(i["status"] in ("ok", "optimal") for i in INFO)
    assert max(i["intervention"] for i in INFO) > 0.05
    ep, er = pose_error_norms(panda.tcp(Q[-1]), Tg)
    assert ep < 5e-3 and er < 2e-2, (ep, er)


def test_cbf_off_path_obstacle_no_intervention(panda):
    Tg = goal_pose(panda, [0.45, 0.35, 0.45])
    bars = [Barrier(SphereField([0.60, -0.30, 0.10], 0.04), 0.01, name="obs"),
            Barrier("table", 0.01, name="table")]
    filt = CBFFilter(panda)
    Qn, QDn, _ = rollout(panda, Tg, 300)
    Q, QD, INFO = rollout(panda, Tg, 300, _cbf(filt, bars))
    assert max(i["intervention"] for i in INFO) < 1e-12
    np.testing.assert_array_equal(Q, Qn)
    assert all(i["status"] == "ok" for i in INFO)          # no QP was needed


def test_cbf_recovers_from_violation(panda):
    q = panda.q_home.copy()
    c, r, _, _ = panda.spheres(q)
    i = panda.sphere_groups["hand"][0]
    n = np.array([0.0, 1.0, 0.0])
    ball = SphereField(c[i] + n * (r[i] + 0.04 + 0.01 - 0.02), 0.04)   # h = -0.02 at start
    bars = [Barrier(ball, 0.01, name="obs")]
    filt = CBFFilter(panda, alpha=5.0)
    h0 = filt.evaluate(q, 0.08, bars)["obs"]
    assert h0 < -0.019
    hs = [h0]
    for _ in range(120):
        qd, info = filt.filter(q, np.zeros(7), 0.08, bars)
        assert info["status"] == "optimal"
        q = q + qd * DT
        hs.append(filt.evaluate(q, 0.08, bars)["obs"])
    hs = np.array(hs)
    assert np.all(np.diff(hs) > -1e-6)
    assert hs[30] > 0.25 * h0 and hs[-1] > -1e-3            # about exp(-alpha t)


def test_joint_limits_never_exceeded(panda):
    filt = CBFFilter(panda, alpha_q=5.0)
    rng = np.random.default_rng(3)
    for trial in range(4):
        q = panda.q_home.copy()
        direction = rng.choice([-1.0, 1.0], size=7)
        for k in range(400):
            qd_nom = direction * panda.qd_max               # push every joint into a limit
            if trial >= 2:
                qd_nom = qd_nom * rng.uniform(-0.2, 1.0, 7)
            qd, info = filt.filter(q, qd_nom, 0.08, [])
            assert np.all(np.abs(qd) <= panda.qd_max + 1e-12)
            q = q + qd * DT
            assert np.all(q >= panda.q_lower) and np.all(q <= panda.q_upper), (trial, k)
        assert np.all(q >= panda.q_lower + filt.joint_buffer - 1e-9)
        assert np.all(q <= panda.q_upper - filt.joint_buffer + 1e-9)
    # without the joint-limit CBF the joints overshoot
    filt_off = CBFFilter(panda, use_joint_limits=False)
    q = panda.q_home.copy()
    for _ in range(400):
        qd, _ = filt_off.filter(q, panda.qd_max, 0.08, [])
        q = q + qd * DT
    assert np.any(q > panda.q_upper)


def test_soft_barrier_is_violated_only_through_slack(panda):
    q = panda.q_home.copy()
    c, r, J, _ = panda.spheres(q)
    i = int(panda.sphere_groups["hand"][0])
    ex = np.array([1.0, 0.0, 0.0])
    m, h0 = 0.01, -0.005
    hard = PlaneField(c[i] - ex * (r[i] + m + h0), ex)      # hard wall behind (needs +x)
    soft = PlaneField(c[i] + ex * (r[i] + m + h0), -ex)     # soft wall in front (needs -x)
    mask = np.zeros(panda.n_spheres, bool)
    mask[i] = True
    bars = [Barrier(hard, m, name="obs", sphere_mask=mask),
            Barrier(soft, m, soft=True, name="unk", sphere_mask=mask)]
    filt = CBFFilter(panda, alpha=5.0)
    qd, info = filt.filter(q, np.zeros(7), 0.08, bars)
    assert info["status"] == "optimal"
    assert info["h_min"]["obs"] == pytest.approx(h0) and info["h_min"]["unk"] == pytest.approx(h0)
    a = J[i].T @ ex
    rate = a @ qd
    assert rate >= -filt.alpha * h0 - 1e-6                  # hard: xdot >= 0.025
    s = info["slack"]["unk"]
    assert s > 0.04                                         # soft: needs s >= 0.05
    assert -rate + s >= -filt.alpha * h0 - 1e-6
    # soft barrier alone: slack = delta / (1 + w_s |a|^2), about 0.1 % of delta
    qd2, info2 = filt.filter(q, np.zeros(7), 0.08, bars[1:])
    s2 = info2["slack"]["unk"]
    assert info2["status"] == "optimal" and s2 < 1e-2 * (-filt.alpha * h0)
    assert -(a @ qd2) + s2 >= -filt.alpha * h0 - 1e-6


def test_soft_barrier_behaves_almost_hard_without_conflict(panda):
    Tg, _ = _obstacle_on_path_setup(panda)
    bars = [Barrier(SphereField([0.45, 0.17, 0.47], 0.04), 0.01, soft=True, name="unk")]
    filt = CBFFilter(panda)
    Q, _, INFO = rollout(panda, Tg, 480, _cbf(filt, bars))
    assert min_h(filt, Q, 0.08, bars)["unk"] >= -H_TOL
    assert max(i["slack"]["unk"] for i in INFO) < 1e-3     # [m/s]


def test_infeasible_hard_constraints_fall_back_gracefully(panda):
    q = panda.q_home.copy()
    c, r, _, _ = panda.spheres(q)
    i = int(panda.sphere_groups["hand"][0])
    ex = np.array([1.0, 0.0, 0.0])
    mask = np.zeros(panda.n_spheres, bool)
    mask[i] = True
    bars = [Barrier(PlaneField(c[i] - ex * (r[i] + 0.005), ex), 0.01, name="a", sphere_mask=mask),
            Barrier(PlaneField(c[i] + ex * (r[i] + 0.005), -ex), 0.01, name="b", sphere_mask=mask)]
    filt = CBFFilter(panda)
    qn = 0.3 * panda.qd_max
    qd, info = filt.filter(q, qn, 0.08, bars)
    assert info["status"] == "relaxed"
    assert np.all(np.isfinite(qd)) and np.all(np.abs(qd) <= panda.qd_max + 1e-12)
    assert info["slack"]["a"] > 0 and info["slack"]["b"] > 0


def test_extra_spheres_of_attached_object(panda):
    """A held object avoids a ball that only the object would hit."""
    Tg = goal_pose(panda, [0.45, 0.35, 0.45])
    ball = SphereField([0.45, 0.17, 0.34], 0.04)
    local = np.array([0.0, 0.0, 0.1034 + 0.05])            # object centre in the hand frame

    def extra_fn(q):
        Th = panda.fk_all(q)["panda_hand"]
        cw = (Th[:3, :3] @ local + Th[:3, 3])[None]
        return cw, np.array([0.03]), panda.jacobian_point(q, "panda_hand", cw)

    filt = CBFFilter(panda)
    bars = [Barrier(ball, 0.01, name="obs")]
    bars_no = [Barrier(ball, 0.01, name="obs", include_extra=False)]
    Qn, _, _ = rollout(panda, Tg, 480, _cbf(filt, bars_no), extra_fn=extra_fn)
    assert min_h(filt, Qn, 0.08, bars_no)["obs"] > 0.0      # the robot itself never gets close
    assert min_h(filt, Qn, 0.08, bars, extra_fn)["obs"] < -0.01
    Q, _, INFO = rollout(panda, Tg, 480, _cbf(filt, bars), extra_fn=extra_fn)
    assert min_h(filt, Q, 0.08, bars, extra_fn)["obs"] >= -H_TOL
    assert pose_error_norms(panda.tcp(Q[-1]), Tg)[0] < 5e-3


def test_sphere_mask_and_exclude(panda):
    q = panda.q_home.copy()
    c, r, _, _ = panda.spheres(q)
    hand = panda.sphere_groups["hand"]
    ball = SphereField(c[hand[0]] + [0.0, r[hand[0]] + 0.04 + 0.005, 0.0], 0.04)
    filt = CBFFilter(panda)
    arm_mask = np.zeros(panda.n_spheres, bool)
    arm_mask[panda.sphere_groups["arm"]] = True
    h_all = filt.evaluate(q, 0.08, [Barrier(ball, 0.01, name="o")])["o"]
    h_arm = filt.evaluate(q, 0.08, [Barrier(ball, 0.01, name="o", sphere_mask=arm_mask)])["o"]
    h_idx = filt.evaluate(q, 0.08, [Barrier(ball, 0.01, name="o",
                                            sphere_mask=panda.sphere_groups["arm"])])["o"]
    assert h_all < 0 < h_arm and h_idx == h_arm
    # excluded spheres are ignored
    h_ex = filt.evaluate(q, 0.08, [Barrier(ball, 0.01, name="o")],
                         exclude_spheres=np.r_[hand, panda.sphere_groups["fingers"]])["o"]
    assert h_ex == pytest.approx(h_arm)
    # table margin None -> filter default
    hb = barrier_values(Barrier("table"), c, r, table_margin=0.02)[0]
    np.testing.assert_allclose(hb, c[:, 2] - r - 0.02)


@pytest.fixture(scope="module")
def clutter_fields():
    grid = VoxelGrid()
    C = grid.centers()

    def box(lo, hi):
        return np.all((C >= lo) & (C <= hi), axis=-1)
    obs = box([0.42, 0.05, 0.0], [0.52, 0.12, 0.12]) | box([0.42, -0.13, 0.0], [0.52, -0.06, 0.10])
    unk = box([0.56, -0.20, 0.0], [0.70, 0.20, 0.10])
    return DistanceField.from_mask(obs, grid), DistanceField.from_mask(unk, grid)


@pytest.mark.parametrize("yaw,hit", [(0.3, "obs"), (np.pi / 2, "unk")])
def test_cbf_with_voxel_distance_field(panda, clutter_fields, yaw, hit):
    """Descent between two voxel blocks: the CBF keeps clear of the blocks and the unknown space."""
    f_obs, f_unk = clutter_fields
    Tg = goal_pose(panda, [0.47, 0.0, 0.05], yaw=yaw)
    bars = [Barrier(f_obs, 0.01, name="obs"), Barrier("table", 0.01, name="table"),
            Barrier(f_unk, 0.01, soft=True, name="unk")]
    filt = CBFFilter(panda)
    Qn, _, _ = rollout(panda, Tg, 420)
    assert min_h(filt, Qn, 0.08, bars)[hit] < -0.015        # nominal violates
    Q, _, INFO = rollout(panda, Tg, 420, _cbf(filt, bars))
    h = min_h(filt, Q, 0.08, bars)
    assert h["obs"] >= -H_TOL and h["table"] >= -H_TOL and h["unk"] >= -H_TOL, h
    assert all(i["status"] in ("ok", "optimal") for i in INFO)


def test_cbf_solve_time(panda, clutter_fields):
    """One step with about 60 spheres and 3 barriers solves in under 1.5 ms."""
    f_obs, f_unk = clutter_fields
    T = goal_pose(panda, [0.47, 0.0, 0.10])
    q, ok = panda.ik(T, panda.q_home)
    assert ok
    arm_hand = np.r_[panda.sphere_groups["arm"], panda.sphere_groups["hand"]]
    bars = [Barrier(f_obs, 0.01, name="obs"), Barrier("table", 0.01, name="table", sphere_mask=arm_hand),
            Barrier(f_unk, 0.01, soft=True, name="unk")]
    ec = T[:3, 3] + np.array([[0, 0, -0.02], [0.02, 0, -0.02], [-0.02, 0, -0.02],
                              [0, 0.02, -0.02], [0, -0.02, -0.02]])
    extra = (ec, np.full(5, 0.012), panda.jacobian_point(q, "panda_hand", ec))
    filt = CBFFilter(panda)
    qn = nominal_qd(panda, q, goal_pose(panda, [0.47, 0.0, 0.04]))
    tot, sol = [], []
    for _ in range(200):
        qd, info = filt.filter(q, qn, 0.08, bars, extra_spheres=extra)
        tot.append(info["total_ms"]); sol.append(info["solve_ms"])
    assert info["status"] == "optimal" and info["n_rows"] >= 30
    assert np.median(sol) < 1.5 and np.median(tot) < 1.5, (np.median(sol), np.median(tot))


def test_cbf_random_scenes_stay_safe(panda):
    rng = np.random.default_rng(7)
    filt = CBFFilter(panda)
    worst = np.inf
    for trial in range(6):
        p0 = np.array([rng.uniform(0.35, 0.6), rng.uniform(-0.25, 0.25), rng.uniform(0.2, 0.4)])
        p1 = np.array([rng.uniform(0.35, 0.6), rng.uniform(-0.25, 0.25), rng.uniform(0.1, 0.3)])
        q0, ok0 = panda.ik(goal_pose(panda, p0), panda.q_home)
        if not ok0:
            continue
        mid = 0.5 * (p0 + p1) + rng.normal(0, 0.02, 3) + [0, 0, 0.03]
        bars = [Barrier(SphereField(mid, rng.uniform(0.02, 0.05)), 0.01, name="obs"),
                Barrier("table", 0.01, name="table")]
        if filt.evaluate(q0, 0.08, bars)["obs"] < 0.01:
            continue
        Q, _, INFO = rollout(panda, goal_pose(panda, p1, yaw=rng.uniform(-1, 1)), 300,
                             _cbf(filt, bars), q0=q0)
        worst = min(worst, min_h(filt, Q, 0.08, bars)["obs"])
    assert worst >= -H_TOL


def test_apf_reduces_violation(panda):
    Tg, bars = _obstacle_on_path_setup(panda)
    filt = CBFFilter(panda)
    Qn, _, _ = rollout(panda, Tg, 480)
    h_nom = min_h(filt, Qn, 0.08, bars)["obs"]

    def apf(q, qn, extra):
        return apf_qd(panda, q, qn, 0.08, bars, APFParams(), extra_spheres=extra), {}
    Q, QD, _ = rollout(panda, Tg, 480, apf)
    h_apf = min_h(filt, Q, 0.08, bars)["obs"]
    assert h_apf > h_nom + 0.03, (h_nom, h_apf)
    assert np.all(np.abs(QD) <= panda.qd_max + 1e-12)


def test_apf_inactive_far_from_obstacles(panda):
    q = panda.q_home.copy()
    qn = nominal_qd(panda, q, goal_pose(panda, [0.45, 0.2, 0.4]))
    bars = [Barrier(SphereField([0.6, -0.3, 0.1], 0.03), 0.01, name="obs"),
            Barrier("table", 0.01, name="table")]
    qd, info = apf_qd(panda, q, qn, 0.08, bars, APFParams(), return_info=True)
    np.testing.assert_array_equal(qd, qn)
    assert info["n_repulsive"] == 0
    # dict form {name: (field, margin)} is accepted too
    qd2 = apf_qd(panda, q, qn, 0.08, {"obs": (bars[0].field, 0.01)}, APFParams())
    np.testing.assert_array_equal(qd2, qn)


def test_apf_repulsion_direction(panda):
    """APF pushes the sphere away from the obstacle."""
    q = panda.q_home.copy()
    c, r, J, _ = panda.spheres(q)
    i = int(panda.sphere_groups["hand"][0])
    ball = SphereField(c[i] + [0.0, r[i] + 0.04 + 0.03, 0.0], 0.04)   # clearance 3 cm < d0
    mask = np.zeros(panda.n_spheres, bool)
    mask[i] = True
    bars = [Barrier(ball, 0.0, name="obs", sphere_mask=mask)]
    qd = apf_qd(panda, q, np.zeros(7), 0.08, bars, APFParams())
    _, g = ball.query(c[i:i + 1])
    assert g[0] @ (J[i] @ qd) > 0


def test_quadprog_fallback_matches_daqp(panda, monkeypatch):
    import safegrasp.control.cbf as cbf_mod
    Tg, bars = _obstacle_on_path_setup(panda)
    filt = CBFFilter(panda)
    q = panda.q_home.copy()
    for _ in range(40):                                     # move until the ball constraint binds
        qn = nominal_qd(panda, q, Tg)
        qd, info = filt.filter(q, qn, 0.08, bars)
        if info["status"] == "optimal":
            break
        q = q + qd * DT
    assert info["status"] == "optimal" and info["solver"] == "daqp"
    monkeypatch.setattr(cbf_mod, "_daqp", None)
    qd2, info2 = filt.filter(q, qn, 0.08, bars)
    assert info2["solver"] == "quadprog" and info2["status"] == "optimal"
    np.testing.assert_allclose(qd2, qd, atol=1e-6)
    for key in ("h_min", "n_rows", "n_active", "slack", "status", "solve_ms", "total_ms",
                "intervention"):
        assert key in info2
