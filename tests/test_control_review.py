"""Edge-case and regression tests for safegrasp.control."""
from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.spatial.transform import Rotation

import safegrasp.control.cbf as cbf_mod
from safegrasp.control.apf import APFParams, apf_qd
from safegrasp.control.cbf import (Barrier, CBFFilter, barrier_values, robot_sphere_set)
from safegrasp.control.nominal import nominal_qd, pose_error, saturate
from safegrasp.kinematics.panda import Panda, rotz
from safegrasp.perception.voxel import DistanceField, VoxelGrid

DT = 1.0 / 60.0
H_TOL = 2e-3


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


class UnionField:
    """Distance to a union of regions: the minimum over its members."""

    def __init__(self, fields):
        self.fields = fields

    def query(self, p):
        out = [f.query(p) for f in self.fields]
        d = np.stack([o[0] for o in out])
        g = np.stack([o[1] for o in out])
        k = np.argmin(d, axis=0)
        i = np.arange(d.shape[1])
        return d[k, i], g[k, i]


@pytest.fixture(scope="module")
def panda():
    return Panda()


def topdown(panda, xyz, yaw=0.0):
    T = panda.tcp(panda.q_home).copy()
    T[:3, :3] = rotz(yaw) @ T[:3, :3]
    T[:3, 3] = xyz
    return T


def fd_rows(panda, q, bar, idx, width=0.08, eps=1e-6, table_margin=0.01):
    """Finite-difference dh_i/dq, shape (n, 7), for spheres idx of barrier bar."""
    r = robot_sphere_set(panda, q, width).radii[idx]
    J = np.zeros((len(idx), 7))
    for j in range(7):
        dq = np.zeros(7)
        dq[j] = eps
        hp, _ = barrier_values(bar, robot_sphere_set(panda, q + dq, width).centers[idx], r, table_margin)
        hm, _ = barrier_values(bar, robot_sphere_set(panda, q - dq, width).centers[idx], r, table_margin)
        J[:, j] = (hp - hm) / (2 * eps)
    return J


def one_hot(panda, *idx):
    m = np.zeros(panda.n_spheres, bool)
    m[list(idx)] = True
    return m


def test_cbf_rows_match_finite_differences_analytic(panda):
    """QP rows equal dh/dq and b = -alpha h, with per-barrier alpha."""
    rng = np.random.default_rng(0)
    filt = CBFFilter(panda, alpha=4.0, activation_dist=10.0)
    bars = [Barrier(SphereField([0.5, 0.1, 0.3], 0.05), 0.01, name="ball"),
            Barrier("table", name="table", alpha=2.0),
            Barrier(PlaneField([0.6, 0.0, 0.0], [-1.0, 0.2, 0.1]), 0.02, name="plane")]
    for _ in range(4):
        q = panda.q_home + rng.normal(0, 0.3, 7)
        rows, _, _ = filt.build_rows(robot_sphere_set(panda, q, 0.08), bars)
        assert len(rows) == 3
        for row, bar in zip(rows, bars):
            FD = fd_rows(panda, q, bar, row["sphere"])
            np.testing.assert_allclose(row["A"], FD, atol=1e-7 * max(1.0, np.abs(FD).max()))
            alpha = filt.alpha if bar.alpha is None else bar.alpha
            np.testing.assert_allclose(row["b"], -alpha * row["h"], rtol=1e-12)


def test_cbf_rows_match_finite_differences_voxel(panda):
    """Voxel distance-field rows agree with dh/dq."""
    grid = VoxelGrid()
    C = grid.centers()
    mask = np.all((C >= [0.42, 0.05, 0.0]) & (C <= [0.52, 0.12, 0.12]), axis=-1)
    bar = Barrier(DistanceField.from_mask(mask, grid), 0.01, name="vox")
    filt = CBFFilter(panda)
    rng = np.random.default_rng(1)
    n_checked = 0
    for _ in range(12):
        T = topdown(panda, [rng.uniform(0.42, 0.52), rng.uniform(-0.02, 0.18), rng.uniform(0.12, 0.25)],
                    yaw=rng.uniform(-1, 1))
        q, ok = panda.ik(T, panda.q_home)
        if not ok:
            continue
        rows, _, _ = filt.build_rows(robot_sphere_set(panda, q, 0.08), [bar])
        if not rows:
            continue
        A = rows[0]["A"]
        FD = fd_rows(panda, q, bar, rows[0]["sphere"], eps=1e-5)
        keep = np.linalg.norm(FD, axis=1) > 1e-3
        cos = np.einsum("ij,ij->i", A[keep], FD[keep]) / (
            np.linalg.norm(A[keep], axis=1) * np.linalg.norm(FD[keep], axis=1))
        assert np.all(cos > 0.97), cos.min()
        ratio = np.linalg.norm(A[keep], axis=1) / np.linalg.norm(FD[keep], axis=1)
        assert np.all((ratio > 0.85) & (ratio < 1.15)), ratio
        n_checked += int(keep.sum())
    assert n_checked > 30


def _osqp_reference(qd_nom, lb, ub, A, b, soft, reg, w_s):
    """Solve the same QP independently with OSQP, one slack per soft row."""
    import osqp
    m = len(b)
    si = np.flatnonzero(soft)
    ns = len(si)
    n = 7 + ns
    P = np.diag(np.r_[np.full(7, 2.0 * (1.0 + reg)), np.full(ns, 2.0 * w_s)])
    qv = np.r_[-2.0 * qd_nom, np.zeros(ns)]
    G = np.zeros((m, n))
    G[:, :7] = A
    G[si, 7 + np.arange(ns)] = 1.0
    Aall = np.vstack([G, np.eye(n)])
    lo = np.r_[b, lb, np.zeros(ns)]
    hi = np.r_[np.full(m, np.inf), ub, np.full(ns, np.inf)]
    solver = osqp.OSQP()
    solver.setup(P=sp.csc_matrix(P), q=qv, A=sp.csc_matrix(Aall), l=lo, u=hi, eps_abs=1e-10,
                 eps_rel=1e-10, polish=True, verbose=False, max_iter=200000)
    res = solver.solve()
    assert res.info.status in ("solved", "solved inaccurate"), res.info.status
    return res.x[:7]


def test_qp_solution_matches_independent_osqp(panda):
    rng = np.random.default_rng(5)
    filt = CBFFilter(panda)
    n_cmp = 0
    for trial in range(40):
        T = topdown(panda, [rng.uniform(0.35, 0.6), rng.uniform(-0.25, 0.25), rng.uniform(0.15, 0.45)],
                    yaw=rng.uniform(-1.5, 1.5))
        q, ok = panda.ik(T, panda.q_home)
        if not ok:
            continue
        c, r, _, _ = panda.spheres(q)
        bars = []
        for k in range(rng.integers(1, 4)):
            i = rng.choice(np.r_[panda.sphere_groups["hand"], panda.sphere_groups["fingers"]])
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            R = rng.uniform(0.02, 0.06)
            gap = rng.uniform(-0.015, 0.03)                         # h of sphere i
            bars.append(Barrier(SphereField(c[i] + u * (r[i] + 0.01 + gap + R), R), 0.01,
                                soft=bool(rng.random() < 0.4), name=f"o{k}"))
        qd_nom = rng.uniform(-0.5, 0.5, 7) * panda.qd_max
        qd, info = filt.filter(q, qd_nom, 0.08, bars)
        if info["status"] != "optimal":
            continue
        rows, _, _ = filt.build_rows(robot_sphere_set(panda, q, 0.08), bars)
        A = np.concatenate([rw["A"] for rw in rows])
        b = np.concatenate([rw["b"] for rw in rows])
        soft = np.concatenate([np.full(len(rw["b"]), rw["soft"]) for rw in rows])
        lb, ub = filt.velocity_bounds(q)
        ref = _osqp_reference(qd_nom, lb, ub, A, b, soft, filt.reg, filt.slack_weight)
        np.testing.assert_allclose(qd, ref, atol=2e-5)
        n_cmp += 1
    assert n_cmp >= 10


def test_velocity_bounds_are_the_joint_limit_cbf(panda):
    """The velocity box is the joint-limit CBF and brings joints beyond a limit back inside."""
    filt = CBFFilter(panda, alpha_q=5.0)
    q = panda.q_home.copy()
    lb, ub = filt.velocity_bounds(q)
    np.testing.assert_allclose(lb, np.maximum(-panda.qd_max, -5.0 * (q - panda.q_lower - 0.02)))
    np.testing.assert_allclose(ub, np.minimum(panda.qd_max, 5.0 * (panda.q_upper - 0.02 - q)))
    # joint 1 above its upper limit, joint 4 below its lower limit
    q = panda.q_home.copy()
    q[0] = panda.q_upper[0] + 0.1
    q[3] = panda.q_lower[3] - 0.1
    lb, ub = filt.velocity_bounds(q)
    assert np.all(lb <= ub) and ub[0] < 0 and lb[3] > 0
    for _ in range(300):
        qd, info = filt.filter(q, np.full(7, 1.0) * panda.qd_max * np.r_[1, 0, 0, -1, 0, 0, 0],
                               0.08, [])
        assert np.all(np.isfinite(qd)) and np.all(np.abs(qd) <= panda.qd_max + 1e-12)
        q = q + qd * DT
    assert panda.q_upper[0] - 0.02 - 1e-3 <= q[0] <= panda.q_upper[0]
    assert panda.q_lower[3] <= q[3] <= panda.q_lower[3] + 0.02 + 1e-3


def test_pose_error_rotvec_is_world_angular_velocity(panda):
    """exp(e_r) R_cur = R_goal, and e_r used as a world angular velocity converges."""
    rng = np.random.default_rng(2)
    for ang in [0.1, 1.0, 2.5, np.pi - 1e-3, np.pi - 1e-6]:
        for _ in range(5):
            Rc = Rotation.random(random_state=rng).as_matrix()
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            Rg = Rotation.from_rotvec(ang * ax).as_matrix() @ Rc
            Tc, Tg = np.eye(4), np.eye(4)
            Tc[:3, :3], Tg[:3, :3] = Rc, Rg
            e = pose_error(Tc, Tg)[3:]
            np.testing.assert_allclose(Rotation.from_rotvec(e).as_matrix() @ Rc, Rg, atol=1e-6)
            errs = []
            R = Rc
            for _ in range(400):
                Tc[:3, :3] = R
                er = pose_error(Tc, Tg)[3:]
                errs.append(np.linalg.norm(er))
                w = saturate(2.0 * er, 1.0)
                R = Rotation.from_rotvec(w * DT).as_matrix() @ R      # world-frame angular velocity
            assert errs[-1] < 1e-3 and np.all(np.diff(errs) < 1e-12)


def test_nominal_at_singularity(panda):
    """At q = 0 (rank-5 J) the command stays bounded and the posture term moves no TCP."""
    q = np.zeros(7)
    J = panda.jacobian_tcp(q)
    assert np.linalg.svd(J, compute_uv=False)[-1] < 1e-9
    Tg = panda.tcp(q)
    Tg[:3, 3] += [0.0, 0.05, -0.05]
    post = q + np.array([0.3, -0.2, 0.4, -0.5, 0.2, 0.6, -0.3])
    qd, info = nominal_qd(panda, q, Tg, q_posture=post, return_info=True)
    assert np.all(np.isfinite(qd)) and np.all(np.abs(qd) <= panda.qd_max + 1e-12)
    assert np.linalg.norm(info["qd_null"]) > 1e-3
    assert np.linalg.norm(J @ info["qd_null"]) < 1e-9


def _violated_ball(panda, q, depth=0.02):
    c, r, _, _ = panda.spheres(q)
    i = int(panda.sphere_groups["hand"][0])
    return i, SphereField(c[i] + np.array([0.0, 1.0, 0.0]) * (r[i] + 0.01 - depth + 0.04), 0.04)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_filter_nonfinite_nominal_is_replaced_by_stop(panda, bad):
    """A non-finite nominal is replaced by zero, which is then filtered."""
    q = panda.q_home.copy()
    filt = CBFFilter(panda)
    qn = 0.3 * panda.qd_max.copy()
    qn[2] = bad
    lb, ub = filt.velocity_bounds(q)
    for bars in ([], [Barrier("table")]):
        qd, info = filt.filter(q, qn, 0.08, bars)
        np.testing.assert_array_equal(qd, np.zeros(7))
        assert info["nominal_ok"] is False and info["status"] == "ok"
    # with a violated barrier the zero command still pushes the sphere out
    i, ball = _violated_ball(panda, q)
    bars = [Barrier(ball, 0.01, name="obs")]
    qd, info = filt.filter(q, qn, 0.08, bars)
    assert np.all(np.isfinite(qd)) and np.all(qd >= lb - 1e-9) and np.all(qd <= ub + 1e-9)
    assert info["status"] == "optimal" and not info["nominal_ok"]
    rows, _, _ = filt.build_rows(robot_sphere_set(panda, q, 0.08), bars)
    assert np.all(rows[0]["A"] @ qd >= rows[0]["b"] - 1e-6)
    assert np.isfinite(info["intervention"])


@pytest.mark.parametrize("what", ["q_nan", "q_inf", "width_nan"])
def test_filter_nonfinite_state_stops(panda, what):
    """A non-finite q or width gives zero velocity, status 'failed' and NaN h_min."""
    q, w = panda.q_home.copy(), 0.08
    if what == "q_nan":
        q[4] = np.nan
    elif what == "q_inf":
        q[1] = np.inf
    else:
        w = float("nan")
    filt = CBFFilter(panda)
    bars = [Barrier("table"), Barrier(SphereField([0.45, 0.0, 0.3], 0.03), 0.01, name="obs", soft=True)]
    qd, info = filt.filter(q, 0.2 * panda.qd_max, w, bars)
    np.testing.assert_array_equal(qd, np.zeros(7))
    assert info["status"] == "failed"
    assert all(np.isnan(v) for v in info["h_min"].values()) and set(info["h_min"]) == {"table", "obs"}
    assert all(np.isnan(v) for v in filt.evaluate(q, w, bars).values())


def test_nonfinite_extra_spheres_and_table_do_not_produce_nan(panda):
    q = panda.q_home.copy()
    c, r, J, _ = panda.spheres(q)
    h, g = barrier_values(Barrier("table"), np.array([[0.0, 0.0, np.nan], [0.0, 0.0, 0.2]]),
                          np.array([0.01, 0.01]))
    assert h[0] == np.inf and np.all(g[0] == 0) and h[1] == pytest.approx(0.18)
    extra = (np.array([[np.nan, 0.0, 0.3]]), np.array([0.02]), np.zeros((1, 3, 7)))
    filt = CBFFilter(panda)
    bars = [Barrier("table"), Barrier(SphereField([0.45, 0.2, 0.4], 0.03), 0.01, name="obs")]
    qd, info = filt.filter(q, 0.2 * panda.qd_max, 0.08, bars, extra_spheres=extra)
    assert np.all(np.isfinite(qd)) and all(np.isfinite(v) for v in info["h_min"].values())


def test_nominal_nonfinite_inputs_return_zero(panda):
    Tg = panda.tcp(panda.q_home)
    q_bad = panda.q_home.copy()
    q_bad[3] = np.nan
    T_bad = Tg.copy()
    T_bad[0, 3] = np.inf
    for args, kw in [((q_bad, Tg), {}), ((panda.q_home, T_bad), {}),
                     ((panda.q_home, Tg), dict(q_posture=np.full(7, np.nan)))]:
        qd = nominal_qd(panda, *args, **kw)
        np.testing.assert_array_equal(qd, np.zeros(7))
        qd, info = nominal_qd(panda, *args, return_info=True, **kw)
        assert info["valid"] is False and np.all(qd == 0)
    # NaN posture is irrelevant when the posture task is off
    qd, info = nominal_qd(panda, panda.q_home, Tg, q_posture=np.full(7, np.nan), k_null=0.0,
                          return_info=True)
    assert info["valid"] and np.all(np.isfinite(qd))


def test_apf_nonfinite_inputs(panda):
    q = panda.q_home.copy()
    i, ball = _violated_ball(panda, q)
    bars = [Barrier(ball, 0.01, name="obs")]
    qd = apf_qd(panda, q, np.full(7, np.nan), 0.08, bars, APFParams())
    assert np.all(np.isfinite(qd)) and np.all(np.abs(qd) <= panda.qd_max + 1e-12)
    c, r, J, _ = panda.spheres(q)
    _, g = ball.query(c[i:i + 1])
    assert g[0] @ (J[i] @ qd) > 0                          # repulsion still pushes out
    q[0] = np.nan
    qd, info = apf_qd(panda, q, np.zeros(7), 0.08, bars, APFParams(), return_info=True)
    np.testing.assert_array_equal(qd, np.zeros(7))
    assert info["valid"] is False


def _two_sphere_setup(panda):
    """Sphere A (hand) and sphere B (top arm sphere, far from A) at q_home."""
    q = panda.q_home.copy()
    c, r, J, _ = panda.spheres(q)
    A = int(panda.sphere_groups["hand"][0])
    arm = panda.sphere_groups["arm"]
    B = int(arm[np.argmax(c[arm, 2])])
    assert np.linalg.norm(c[A] - c[B]) > 0.3
    return q, c, r, J, A, B


def test_soft_slack_does_not_leak_between_spheres(panda):
    """Slack used by sphere A in a soft barrier does not let sphere B enter it."""
    q, c, r, J, A, B = _two_sphere_setup(panda)
    ex, ey = np.eye(3)[0], np.eye(3)[1]
    m, h0, R = 0.01, -0.02, 0.05
    unk = UnionField([SphereField(c[A] + ex * (r[A] + m + h0 + R), R),     # A inside, h = -0.02
                      SphereField(c[B] + ey * (r[B] + m + R), R)])         # B on boundary, h = 0
    wall = PlaneField(c[A] - ex * (r[A] + m + h0), ex)                     # A cannot back out
    bars = [Barrier(wall, m, name="obs", sphere_mask=one_hot(panda, A)),
            Barrier(unk, m, soft=True, name="unk", sphere_mask=one_hot(panda, A, B))]
    aB = J[B].T @ ey
    qd_nom = aB / (aB @ aB) * 0.1                                        # B -> +y at 0.1 m/s
    qd, info = CBFFilter(panda).filter(q, qd_nom, 0.08, bars)
    assert info["status"] == "optimal" and info["slack"]["unk"] > 0.15
    assert ex @ J[A] @ qd >= 0.1 - 1e-6
    assert ey @ J[B] @ qd < 1e-3
    qd_sh, info_sh = CBFFilter(panda, slack_per_row=False).filter(q, qd_nom, 0.08, bars)
    assert ey @ J[B] @ qd_sh > 0.05                                      # shared slack lets B enter


def test_relaxed_mode_never_pushes_safe_spheres_into_obstacles(panda):
    """A needs 2 m/s to escape, so the QP is relaxed; B on the boundary must not be pushed in."""
    q, c, r, J, A, B = _two_sphere_setup(panda)
    ex, ey = np.eye(3)[0], np.eye(3)[1]
    m, h0, R = 0.01, -0.02, 0.05
    obs = UnionField([SphereField(c[A] + ex * (r[A] + m + h0 + R), R),
                      SphereField(c[B] + ey * (r[B] + m + R), R)])
    bars = [Barrier(obs, m, name="obs", sphere_mask=one_hot(panda, A, B), alpha=100.0)]
    aB = J[B].T @ ey
    qd_nom = aB / (aB @ aB) * 0.1
    filt = CBFFilter(panda)
    lb, ub = filt.velocity_bounds(q)
    qd, info = filt.filter(q, qd_nom, 0.08, bars)
    assert info["status"] == "relaxed" and info["slack"]["obs"] > 0.5
    assert ey @ J[B] @ qd <= 1e-6
    assert -ex @ J[A] @ qd > 0.3                                         # A escapes (-x) fast
    assert np.all(qd >= lb - 1e-9) and np.all(qd <= ub + 1e-9)


def test_relaxed_squeeze_goes_no_deeper(panda):
    """A sphere between two violated hard walls is pushed deeper into neither."""
    q = panda.q_home.copy()
    c, r, J, _ = panda.spheres(q)
    i = int(panda.sphere_groups["hand"][0])
    ex = np.eye(3)[0]
    mask = one_hot(panda, i)
    bars = [Barrier(PlaneField(c[i] - ex * (r[i] + 0.005), ex), 0.01, name="a", sphere_mask=mask),
            Barrier(PlaneField(c[i] + ex * (r[i] + 0.005), -ex), 0.01, name="b", sphere_mask=mask)]
    filt = CBFFilter(panda)
    qd, info = filt.filter(q, 0.3 * panda.qd_max * np.r_[1, -1, 1, -1, 1, -1, 1], 0.08, bars)
    assert info["status"] == "relaxed"
    assert abs(ex @ J[i] @ qd) < 1e-6
    assert info["slack"]["a"] == pytest.approx(0.025, abs=1e-3)
    assert info["slack"]["b"] == pytest.approx(0.025, abs=1e-3)


def test_relaxed_when_zero_is_outside_the_joint_box(panda):
    """When stage 1 is infeasible too, stage 2 still gives a finite command inside the joint box."""
    q = panda.q_home.copy()
    q[0] = panda.q_upper[0] + 0.05                    # ub[0] < 0 -> qd = 0 not in the box
    c, r, J, ids = panda.spheres(q)
    l1 = np.flatnonzero(ids == 1)
    i = int(l1[np.argmax(np.linalg.norm(J[l1, :, 0], axis=1))])
    t = J[i, :, 0] / np.linalg.norm(J[i, :, 0])       # direction joint 1 moves sphere i
    mask = one_hot(panda, i)
    bars = [Barrier(PlaneField(c[i] - t * (r[i] + 0.005), t), 0.01, name="a", sphere_mask=mask),
            Barrier(PlaneField(c[i] + t * (r[i] + 0.005), -t), 0.01, name="b", sphere_mask=mask)]
    filt = CBFFilter(panda)
    lb, ub = filt.velocity_bounds(q)
    assert ub[0] < 0
    calls = []
    solve = filt._solve

    def spy(*args):
        out = solve(*args)
        calls.append(out[1])
        return out
    filt._solve = spy
    qd, info = filt.filter(q, np.zeros(7), 0.08, bars)
    assert calls == [False, False, True]              # plain QP, stage 1 infeasible, stage 2 ok
    # the box forces motion along -t, so wall 'a' takes the slack
    assert info["status"] == "relaxed" and info["slack"]["a"] > 0.04
    assert np.all(np.isfinite(qd)) and np.all(qd >= lb - 1e-9) and np.all(qd <= ub + 1e-9)
    assert t @ J[i] @ qd < 0


def test_quadprog_fallback_handles_relaxed(panda, monkeypatch):
    q, c, r, J, A, B = _two_sphere_setup(panda)
    ex, ey = np.eye(3)[0], np.eye(3)[1]
    obs = UnionField([SphereField(c[A] + ex * (r[A] + 0.01 - 0.02 + 0.05), 0.05),
                      SphereField(c[B] + ey * (r[B] + 0.01 + 0.05), 0.05)])
    bars = [Barrier(obs, 0.01, name="obs", sphere_mask=one_hot(panda, A, B), alpha=100.0)]
    aB = J[B].T @ ey
    qd_nom = aB / (aB @ aB) * 0.1
    filt = CBFFilter(panda)
    qd, info = filt.filter(q, qd_nom, 0.08, bars)
    assert info["solver"] == "daqp" and info["status"] == "relaxed"
    monkeypatch.setattr(cbf_mod, "_daqp", None)
    qd2, info2 = filt.filter(q, qd_nom, 0.08, bars)
    assert info2["solver"] == "quadprog" and info2["status"] == "relaxed"
    np.testing.assert_allclose(qd2, qd, atol=1e-5)


def test_info_is_json_serialisable_and_barriers_none(panda):
    """info keys are plain str and barriers=None is accepted."""
    q = panda.q_home.copy()
    i, ball = _violated_ball(panda, q)
    filt = CBFFilter(panda)
    qd, info = filt.filter(q, np.zeros(7), 0.08, [Barrier(ball, 0.01, name="u", soft=True),
                                                  Barrier(ball, 0.01, name="o")])
    assert all(type(k) is str for d in ("h_min", "slack", "n_rows_per") for k in info[d])
    json.dumps(info)
    qd, info = filt.filter(q, 0.1 * panda.qd_max, 0.08, None)
    np.testing.assert_array_equal(qd, 0.1 * panda.qd_max)
    assert info["status"] == "ok" and info["n_rows"] == 0


def test_shared_slack_variant_still_available(panda):
    """slack_per_row=False (one slack per soft barrier) still works, for ablations."""
    q = panda.q_home.copy()
    i, ball = _violated_ball(panda, q)
    for per_row in (True, False):
        qd, info = CBFFilter(panda, slack_per_row=per_row).filter(
            q, np.zeros(7), 0.08, [Barrier(ball, 0.01, soft=True, name="unk")])
        assert info["status"] == "optimal" and np.all(np.isfinite(qd))
        assert 0.0 <= info["slack"]["unk"] < 1e-3


def test_start_inside_obstacle_recovers_without_new_violations(panda):
    """Starting 2 cm inside a ball, min h never drops and spheres outside stay outside."""
    q = panda.q_home.copy()
    c, r, _, _ = panda.spheres(q)
    fing = panda.sphere_groups["fingers"]
    low = fing[np.argmin(c[fing, 2] - r[fing])]
    R = 0.06
    ball = SphereField(c[low] - np.array([0, 0, 1.0]) * (r[low] + 0.01 - 0.02 + R), R)
    bars = [Barrier(ball, 0.01, name="obs")]
    filt = CBFFilter(panda)
    sph = robot_sphere_set(panda, q, 0.08)
    h0, _ = barrier_values(bars[0], sph.centers, sph.radii)
    assert h0.min() < -0.019
    safe0 = h0 >= 0
    Tg = topdown(panda, panda.tcp(q)[:3, 3] - [0, 0, 0.2])
    hmins = [h0.min()]
    for _ in range(180):
        qd, info = filt.filter(q, nominal_qd(panda, q, Tg), 0.08, bars)
        assert info["status"] in ("ok", "optimal", "relaxed")
        q = q + qd * DT
        sph = robot_sphere_set(panda, q, 0.08)
        h, _ = barrier_values(bars[0], sph.centers, sph.radii)
        assert np.all(h[safe0] >= -H_TOL)
        hmins.append(h.min())
    hmins = np.array(hmins)
    assert np.all(np.diff(hmins[hmins < -1e-3]) > -1e-5)
    assert hmins[-1] > -1e-3


def test_random_voxel_clutter_closed_loop(panda):
    """Random voxel clutter with unknown shadows: every barrier stays within H_TOL."""
    grid = VoxelGrid()
    C = grid.centers()

    def box(lo, hi):
        return np.all((C >= lo) & (C <= hi), axis=-1)
    rng = np.random.default_rng(11)
    arm_hand = np.r_[panda.sphere_groups["arm"], panda.sphere_groups["hand"]]
    filt = CBFFilter(panda)
    n_run, worst = 0, dict(obs=np.inf, table=np.inf, unk=np.inf)
    while n_run < 6:
        tgt = np.array([rng.uniform(0.4, 0.62), rng.uniform(-0.2, 0.2), rng.uniform(0.03, 0.08)])
        obs = np.zeros(grid.shape, bool)
        unk = np.zeros(grid.shape, bool)
        for _ in range(rng.integers(3, 7)):
            cxy = tgt[:2] + rng.uniform(-0.15, 0.15, 2)
            if np.linalg.norm(cxy - tgt[:2]) < 0.06:
                continue
            half, hgt = rng.uniform(0.015, 0.04, 2), rng.uniform(0.04, 0.16)
            obs |= box([*(cxy - half), 0.0], [*(cxy + half), hgt])
            unk |= box([cxy[0] - half[0] - 0.06, cxy[1] - half[1], 0.0],
                       [cxy[0] - half[0], cxy[1] + half[1], 0.8 * hgt])
        unk &= ~obs
        bars = [Barrier(DistanceField.from_mask(obs, grid), 0.01, name="obs"),
                Barrier("table", 0.01, name="table", sphere_mask=arm_hand),
                Barrier(DistanceField.from_mask(unk, grid), 0.01, soft=True, name="unk")]
        if min(filt.evaluate(panda.q_home, 0.08, bars).values()) < 0:
            continue
        Tg = topdown(panda, tgt, yaw=rng.uniform(-1.5, 1.5))
        q = panda.q_home.copy()
        for _ in range(300):
            qd, info = filt.filter(q, nominal_qd(panda, q, Tg), 0.08, bars)
            assert info["status"] != "failed" and np.all(np.isfinite(qd))
            q = q + qd * DT
            for k, v in filt.evaluate(q, 0.08, bars).items():
                worst[k] = min(worst[k], v)
        n_run += 1
    assert worst["obs"] >= -H_TOL and worst["table"] >= -H_TOL and worst["unk"] >= -H_TOL, worst
