"""Unit tests for safegrasp.kinematics.panda."""
import json
import os
import time

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from safegrasp.kinematics.panda import (LINK_NAMES, LINK_INDEX, TCP_OFFSET, FINGER_Z, MAX_WIDTH, Panda,
                                        matrix_to_quat, pose_error, quat_to_matrix, rot_log, rotz)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "isaac_franka_fk.npz")
KIN_DIR = os.path.join(os.path.dirname(HERE), "safegrasp", "kinematics")
SPHERES_JSON = os.path.join(KIN_DIR, "panda_spheres.json")
GRIPPER_JSON = os.path.join(KIN_DIR, "panda_gripper_geometry.json")
PROTRUSION_TOL = 0.005        # m, max hull protrusion outside the sphere union


@pytest.fixture(scope="module")
def pd():
    return Panda()


@pytest.fixture(scope="module")
def gt():
    if not os.path.exists(FIXTURE):
        pytest.skip("Isaac ground-truth fixture missing (run scripts/verify_fk_isaac.py --save)")
    d = np.load(FIXTURE)
    return {k: d[k] for k in d.files}


@pytest.fixture(scope="module")
def hulls():
    with open(SPHERES_JSON) as f:
        data = json.load(f)
    return {k: trimesh.convex.convex_hull(np.asarray(v, float)) for k, v in data["hulls"].items()}


def _hull_points(h, n=4000, seed=0):
    pts, _ = trimesh.sample.sample_surface_even(h, n, seed=seed)
    return np.vstack([pts, h.vertices])


def _gap(P, c, r):
    """Return how far (m) the worst point of P sticks out of the sphere union."""
    return float((np.linalg.norm(P[:, None, :] - c[None], axis=-1) - r[None]).min(1).max())


def test_fk_matches_isaac(pd, gt):
    Q = gt["q"].astype(float)
    names = [str(n) for n in gt["body_names"]]
    P, R = gt["pos"].astype(float), quat_to_matrix(gt["quat"].astype(float))
    TL = pd.link_transforms(Q[:, :7], width=2 * Q[:, 7])      # left finger uses finger_joint1
    TR = pd.link_transforms(Q[:, :7], width=2 * Q[:, 8])      # right finger uses finger_joint2
    for bi, name in enumerate(names):
        k = LINK_INDEX[name]
        T = TR[:, k] if name == "panda_rightfinger" else TL[:, k]
        ep = np.linalg.norm(T[:, :3, 3] - P[:, bi], axis=1).max()
        er = max(np.linalg.norm(rot_log(T[n, :3, :3].T @ R[n, bi])) for n in range(len(Q)))
        assert ep < 1e-4 and er < 1e-3, (name, ep, er)
    fk = pd.fk_all(Q[3, :7])
    assert set(fk) == set(LINK_NAMES)
    Th = fk["panda_hand"]
    assert np.allclose(pd.tcp(Q[3, :7]), Th @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, TCP_OFFSET], [0, 0, 0, 1.0]]))


def test_width_is_twice_finger_joint(gt):
    """Each finger origin sits at (0, +-q_f, FINGER_Z) in the hand frame."""
    names = [str(n) for n in gt["body_names"]]
    ih, il, ir = names.index("panda_hand"), names.index("panda_leftfinger"), names.index("panda_rightfinger")
    R = quat_to_matrix(gt["quat"][:, ih].astype(float))
    for idx, j, sign in ((il, 7, 1.0), (ir, 8, -1.0)):
        d = np.einsum("nji,nj->ni", R, gt["pos"][:, idx].astype(float) - gt["pos"][:, ih].astype(float))
        assert np.allclose(d[:, 0], 0.0, atol=1e-5)
        assert np.allclose(d[:, 1], sign * gt["q"][:, j], atol=1e-5)
        assert np.allclose(d[:, 2], FINGER_Z, atol=1e-5)


def test_jacobians_match_isaac(pd, gt):
    names = [str(n) for n in gt["body_names"]]
    for n in range(0, len(gt["q"]), 10):
        q = gt["q"][n, :7].astype(float)
        T = pd.link_transforms(q, 2 * float(gt["q"][n, 7]))
        for bi, name in enumerate(names):
            if bi == 0:
                continue
            Ji = gt["jac"][n, bi - 1, :, :7].astype(float)
            Jl = pd.jacobian_point(q, name, gt["com"][n, bi].astype(float))
            Ja = T[1:8, :3, 2].T * (np.arange(7) < pd._n_joints_of_link[LINK_INDEX[name]])
            assert np.abs(Jl - Ji[:3]).max() < 1e-4 and np.abs(Ja - Ji[3:]).max() < 1e-4, name


def test_limits_and_home(pd, gt):
    assert pd.joint_names == [f"panda_joint{i}" for i in range(1, 8)]
    assert np.allclose(pd.q_lower, gt["limits"][:7, 0], atol=1e-6)
    assert np.allclose(pd.q_upper, gt["limits"][:7, 1], atol=1e-6)
    assert np.allclose(pd.qd_max, 0.8 * np.array([2.175] * 4 + [2.61] * 3))
    assert np.allclose(pd.qd_max / 0.8, gt["vel_limits"][:7], atol=1e-5)
    assert pd.in_limits(pd.q_home)
    T = pd.tcp(pd.q_home)
    assert np.allclose(T[:3, 3], [0.45, 0.0, 0.45], atol=1e-3)
    assert np.allclose(T[:3, 2], [0, 0, -1], atol=1e-3)          # top-down: hand z points down
    assert np.allclose(T[:3, 0], [1, 0, 0], atol=1e-3)
    quat = matrix_to_quat(T[:3, :3])
    assert min(np.abs(quat - [0, 1, 0, 0]).max(), np.abs(quat + [0, 1, 0, 0]).max()) < 1e-3


def test_batched_fk_equals_single(pd):
    rng = np.random.default_rng(0)
    Q, W = pd.random_q(rng, 6), rng.uniform(0, MAX_WIDTH, 6)
    TB = pd.link_transforms(Q, W)
    for n in range(6):
        assert np.allclose(TB[n], pd.link_transforms(Q[n], W[n]), atol=1e-12)
    assert np.allclose(pd.tcp(Q)[2], pd.tcp(Q[2]))
    c, *_ = pd.spheres(Q[4], W[4])
    assert np.allclose(pd.sphere_centers(Q, W)[4], c, atol=1e-12)


def test_rotation_helpers():
    rng = np.random.default_rng(1)
    Rs = Rotation.random(200, random_state=1)
    # the log map is delicate at and near pi
    axes = rng.normal(size=(20, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    near_pi = Rotation.from_rotvec(axes * (np.pi - np.array([0, 1e-9, 1e-6, 1e-3] * 5))[:, None])
    for Rot in list(Rs) + list(near_pi):
        R = Rot.as_matrix()
        assert np.allclose(Rotation.from_rotvec(rot_log(R)).as_matrix(), R, atol=1e-7)
        q = matrix_to_quat(R)
        assert np.allclose(quat_to_matrix(q), R, atol=1e-10) and q[0] >= 0
    T1, T2 = np.eye(4), np.eye(4)
    T2[:3, :3], T2[:3, 3] = rotz(0.3), [0.1, 0.2, 0.3]
    assert np.allclose(pose_error(T1, T2), [0.1, 0.2, 0.3, 0, 0, 0.3])


def test_jacobian_point_finite_difference(pd):
    rng = np.random.default_rng(2)
    h = 1e-6
    for _ in range(5):
        q = pd.random_q(rng)
        for link in ["panda_link2", "panda_link4", "panda_link7", "panda_hand", "panda_leftfinger"]:
            k = LINK_INDEX[link]
            p_loc = np.append(rng.normal(scale=0.1, size=3), 1.0)
            p = (pd.link_transforms(q)[k] @ p_loc)[:3]
            J = pd.jacobian_point(q, link, p)
            Jfd = np.stack([((pd.link_transforms(q + h * e)[k] @ p_loc)[:3]
                             - (pd.link_transforms(q - h * e)[k] @ p_loc)[:3]) / (2 * h) for e in np.eye(7)], 1)
            assert np.abs(J - Jfd).max() < 1e-7, link
        P = rng.normal(scale=0.5, size=(4, 3))
        JB = pd.jacobian_point(q, "panda_link5", P)
        assert JB.shape == (4, 3, 7)
        assert np.allclose(JB, np.stack([pd.jacobian_point(q, "panda_link5", p) for p in P]))


def test_jacobian_tcp_finite_difference(pd):
    rng = np.random.default_rng(3)
    h = 1e-6
    for _ in range(10):
        q = pd.random_q(rng)
        J, T0 = pd.jacobian_tcp(q), pd.tcp(q)
        Jfd = np.zeros((6, 7))
        for j, e in enumerate(np.eye(7)):
            Tp, Tm = pd.tcp(q + h * e), pd.tcp(q - h * e)
            Jfd[:3, j] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
            Jfd[3:, j] = rot_log(Tp[:3, :3] @ Tm[:3, :3].T) / (2 * h)   # world-frame angular velocity
        assert np.abs(J - Jfd).max() < 1e-7
        assert np.allclose(pd._tcp_and_jac(q)[0], T0)


def test_sphere_model_structure(pd):
    c, r, J, ids = pd.spheres(pd.q_home, 0.05)
    S = pd.n_spheres
    assert 20 <= S <= 60
    assert c.shape == (S, 3) and r.shape == (S,) and J.shape == (S, 3, 7) and ids.shape == (S,)
    assert np.all(r > 0.003)
    assert np.all((ids >= 1) & (ids <= 10))                       # links 1..7, hand, fingers
    assert set(np.unique(ids)) == set(range(1, 11))
    g = pd.sphere_groups
    allidx = np.sort(np.concatenate([g["arm"], g["hand"], g["fingers"]]))
    assert np.array_equal(allidx, np.arange(S))
    assert 6 <= len(g["hand"]) <= 8
    assert np.sum(ids == LINK_INDEX["panda_leftfinger"]) == np.sum(ids == LINK_INDEX["panda_rightfinger"]) == 3
    assert np.all(ids[g["arm"]] <= 7) and np.all(ids[g["hand"]] == 8) and np.all(ids[g["fingers"]] >= 9)


def test_sphere_jacobian_finite_difference(pd):
    rng = np.random.default_rng(4)
    h = 1e-6
    for _ in range(5):
        q, w = pd.random_q(rng), rng.uniform(0, MAX_WIDTH)
        _, _, J, _ = pd.spheres(q, w)
        Jfd = np.stack([(pd.spheres(q + h * e, w)[0] - pd.spheres(q - h * e, w)[0]) / (2 * h) for e in np.eye(7)], -1)
        assert np.abs(J - Jfd).max() < 1e-7


def test_finger_spheres_follow_width(pd):
    q = pd.q_home
    c1, _, _, ids = pd.spheres(q, 0.02)
    c2, _, _, _ = pd.spheres(q, 0.08)
    y_hand = pd.fk_all(q)["panda_hand"][:3, 1]
    left, right = ids == LINK_INDEX["panda_leftfinger"], ids == LINK_INDEX["panda_rightfinger"]
    assert np.allclose(c2[left] - c1[left], 0.03 * y_hand)
    assert np.allclose(c2[right] - c1[right], -0.03 * y_hand)
    assert np.allclose(c2[~(left | right)], c1[~(left | right)])


def test_spheres_cover_collision_meshes_link_frame(pd, hulls):
    for link in LINK_NAMES[1:]:
        k = LINK_INDEX[link]
        m = pd.sphere_link_ids == k
        gap = _gap(_hull_points(hulls[link], seed=k), pd._sph_local[m], pd.sphere_radii[m])
        assert gap <= PROTRUSION_TOL, (link, gap)


def test_spheres_cover_collision_meshes_world(pd, hulls):
    rng = np.random.default_rng(5)
    pts_local = {link: _hull_points(hulls[link], 1500, seed=1) for link in LINK_NAMES[1:]}
    for _ in range(3):
        q, w = pd.random_q(rng), rng.uniform(0, MAX_WIDTH)
        T = pd.link_transforms(q, w)
        c, r, _, ids = pd.spheres(q, w)
        for link, P in pts_local.items():
            k = LINK_INDEX[link]
            Pw = P @ T[k, :3, :3].T + T[k, :3, 3]
            assert _gap(Pw, c[ids == k], r[ids == k]) <= PROTRUSION_TOL + 1e-9, link


def test_spheres_speed(pd):
    q = pd.q_home
    best = np.inf
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(200):
            pd.spheres(q, 0.05)
        best = min(best, (time.perf_counter() - t) / 200)
    print(f"spheres(): {best * 1e6:.1f} us/call ({pd.n_spheres} spheres)")
    assert best < 3e-4


def test_ik_random_reachable_targets(pd):
    """ik_multi solves at least 95 % of 200 random reachable poses, quickly."""
    rng = np.random.default_rng(1)
    Qt = pd.random_q(rng, 200)
    Tt = pd.tcp(Qt)
    succ, times = 0, []
    for n in range(200):
        t = time.perf_counter()
        q, ok = pd.ik_multi(Tt[n])
        times.append(time.perf_counter() - t)
        assert pd.in_limits(q)
        if ok:
            e = pose_error(pd.tcp(q), Tt[n])
            assert np.linalg.norm(e[:3]) < 2e-3 and np.linalg.norm(e[3:]) < 2e-2
            succ += 1
    med = float(np.median(times))
    print(f"ik_multi: {succ}/200 success, median {med * 1e3:.2f} ms, mean {np.mean(times) * 1e3:.2f} ms")
    assert succ >= 190
    assert med < 5e-3


def test_ik_topdown_workspace_single_seed(pd):
    """IK seeded from q_home reaches at least 48 of 50 top-down grasp poses."""
    rng = np.random.default_rng(6)
    ok_count = 0
    for _ in range(50):
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ rotz(rng.uniform(-np.pi / 2, np.pi / 2))
        T[:3, 3] = [rng.uniform(0.35, 0.75), rng.uniform(-0.3, 0.3), rng.uniform(0.02, 0.3)]
        q, ok, info = pd.ik(T, pd.q_home, return_info=True)
        ok_count += ok
        assert pd.in_limits(q)
    assert ok_count >= 48


def test_ik_unreachable_reports_failure(pd):
    T = np.eye(4)
    T[:3, 3] = [1.5, 0.0, 0.5]                                     # beyond the ~0.95 m reach
    q, ok = pd.ik_multi(T)
    assert not ok and pd.in_limits(q)


def test_gripper_geometry_json_consistent_with_meshes(pd, hulls):
    """The JSON boxes (TCP frame) contain the hand, link7 and finger hulls."""
    with open(GRIPPER_JSON) as f:
        g = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in json.load(f).items()}
    assert np.isclose(g["tcp_offset"], TCP_OFFSET) and np.isclose(g["max_width"], MAX_WIDTH)
    tol = 2e-5
    for w in (0.0, 0.03, 0.08):
        T = pd.link_transforms(pd.q_home, w)
        T_tcp_inv = np.linalg.inv(pd.tcp(pd.q_home))

        def to_tcp(link):
            M = T_tcp_inv @ T[LINK_INDEX[link]]
            return hulls[link].vertices @ M[:3, :3].T + M[:3, 3]

        for link, lo, hi in (("panda_hand", g["palm_lo"], g["palm_hi"]),
                             ("panda_link7", g["wrist_lo"], g["wrist_hi"])):
            P = to_tcp(link)
            assert np.all(P >= lo - tol) and np.all(P <= hi + tol), link
        inner = w / 2 + g["finger_offset"]
        for link, sgn in (("panda_leftfinger", 1.0), ("panda_rightfinger", -1.0)):
            P = to_tcp(link)
            y = sgn * P[:, 1]                                       # mirrored for the -y finger
            assert np.all(P[:, 0] >= g["finger_x"][0] - tol) and np.all(P[:, 0] <= g["finger_x"][1] + tol)
            assert np.all(P[:, 2] >= g["finger_z"][0] - tol) and np.all(P[:, 2] <= g["finger_z"][1] + tol)
            assert np.all(y >= inner - tol) and np.all(y <= inner + g["finger_thickness"] + tol)
    # TCP lies within the pad, which ends at the fingertip
    assert g["pad_z"][0] < 0.0 < g["pad_z"][1] <= g["finger_z"][1]
    assert abs(g["finger_offset"]) < 1e-3


def test_gripper_geometry_loaded_by_grasping_module():
    try:
        from safegrasp.grasping.gripper import GripperGeometry
    except Exception as e:  # pragma: no cover - grasping module unavailable
        pytest.skip(f"grasping module not importable: {e}")
    geo = GripperGeometry.load(GRIPPER_JSON)
    with open(GRIPPER_JSON) as f:
        d = json.load(f)
    assert os.path.samefile(geo.source, GRIPPER_JSON)
    for k in ("palm_lo", "palm_hi", "wrist_lo", "wrist_hi", "finger_x", "finger_z", "pad_z"):
        assert np.allclose(getattr(geo, k), d[k]), k
    assert np.isclose(geo.finger_thickness, d["finger_thickness"])
    assert np.isclose(geo.finger_offset, d["finger_offset"])
