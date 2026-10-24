"""Edge-case and regression tests for safegrasp.kinematics.panda."""
import json
import os
import time

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from safegrasp.kinematics.panda import (FINGER_Z, LINK_INDEX, LINK_NAMES, MAX_WIDTH, TCP_OFFSET, Panda,
                                        matrix_to_quat, pose_error, quat_to_matrix, rot_log, rotz)

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "isaac_franka_fk.npz")
SPHERES_JSON = os.path.join(os.path.dirname(HERE), "safegrasp", "kinematics", "panda_spheres.json")

# Franka's published modified DH table: rows (a_{i-1}, d_i, alpha_{i-1})
_DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
       (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]


def _mdh(a, d, al, th):
    ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca],
                     [0, 0, 0, 1.0]])


def _fk_dh(q):
    """Reference FK from the DH table: frames of links 0..7, then the hand."""
    T, out = np.eye(4), [np.eye(4)]
    for (a, d, al), th in zip(_DH, q):
        T = T @ _mdh(a, d, al, th)
        out.append(T)
    H = T @ _mdh(0.0, 0.107, 0.0, 0.0)
    H[:3, :3] = H[:3, :3] @ rotz(-np.pi / 4)
    out.append(H)
    return out


@pytest.fixture(scope="module")
def pd():
    return Panda()


@pytest.fixture(scope="module")
def gt():
    if not os.path.exists(FIXTURE):
        pytest.skip("Isaac ground-truth fixture missing")
    d = np.load(FIXTURE)
    return {k: d[k] for k in d.files}


def test_fk_matches_independent_modified_dh(pd):
    rng = np.random.default_rng(10)
    for q in pd.random_q(rng, 200):
        T, Tref = pd.link_transforms(q), _fk_dh(q)
        for k in range(9):                                            # link0..7, hand
            assert np.abs(T[k] - Tref[k]).max() < 1e-12, LINK_NAMES[k]
        tcp_ref = Tref[8][:3, 3] + TCP_OFFSET * Tref[8][:3, 2]
        assert np.abs(pd.tcp(q)[:3, 3] - tcp_ref).max() < 1e-12


def test_fk_matches_isaac_tight(pd, gt):
    """Tighter tolerance than the main FK test; the fixture is float32."""
    Q = gt["q"].astype(float)
    names = [str(n) for n in gt["body_names"]]
    TL = pd.link_transforms(Q[:, :7], width=2 * Q[:, 7])
    TR = pd.link_transforms(Q[:, :7], width=2 * Q[:, 8])
    R = quat_to_matrix(gt["quat"].astype(float))
    for bi, name in enumerate(names):
        T = (TR if name == "panda_rightfinger" else TL)[:, LINK_INDEX[name]]
        assert np.abs(T[:, :3, 3] - gt["pos"][:, bi]).max() < 1e-5, name
        dR = np.einsum("nji,njk->nik", T[:, :3, :3], R[:, bi])      # R_np^T R_isaac
        ang = np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1))
        assert ang.max() < 2e-5, name


def test_isaac_com_inside_collision_hulls(pd, gt):
    """PhysX COMs lie inside each hull and sphere union, so both sit in the right link frames."""
    names = [str(n) for n in gt["body_names"]]
    with open(SPHERES_JSON) as f:
        hulls = json.load(f)["hulls"]
    R = quat_to_matrix(gt["quat"].astype(float))
    for bi, name in enumerate(names):
        h = trimesh.convex.convex_hull(np.asarray(hulls[name], float))
        com = np.einsum("nji,nj->ni", R[:, bi], gt["com"][:, bi].astype(float) - gt["pos"][:, bi])
        nrm = h.face_normals
        b = np.einsum("ij,ij->i", nrm, h.triangles[:, 0])
        assert (b[None] - com @ nrm.T).min() > 0.005, name            # >= 5 mm inside the hull
        if name != "panda_link0":
            m = pd.sphere_link_ids == LINK_INDEX[name]
            gap = (np.linalg.norm(com[:, None] - pd._sph_local[m][None], axis=-1) - pd.sphere_radii[m]).min(1)
            assert gap.max() < 0, name


def test_fk_all_batched_indexes_bodies(pd):
    """fk_all on a batch returns each body for every configuration."""
    rng = np.random.default_rng(11)
    Q = pd.random_q(rng, 12)
    fk = pd.fk_all(Q, 0.03)
    for name in LINK_NAMES:
        assert fk[name].shape == (12, 4, 4), name
        for n in (0, 5, 11):
            assert np.allclose(fk[name][n], pd.fk_all(Q[n], 0.03)[name], atol=1e-14), name


def test_width_clipped_to_physical_range(pd):
    """Width is clamped to [0, 0.08] m, as Isaac clamps each finger joint to [0, 0.04]."""
    q = pd.q_home
    assert np.allclose(pd.spheres(q, 0.12)[0], pd.spheres(q, MAX_WIDTH)[0])
    assert np.allclose(pd.spheres(q, -0.02)[0], pd.spheres(q, 0.0)[0])
    assert np.allclose(pd.link_transforms(q, 1.0), pd.link_transforms(q, MAX_WIDTH))
    Q = np.stack([q, q, q])
    TB = pd.link_transforms(Q, np.array([-0.05, 0.04, 0.3]))
    for T, w in zip(TB, (0.0, 0.04, MAX_WIDTH)):
        assert np.allclose(T, pd.link_transforms(q, w))
    assert np.allclose(pd.sphere_centers(Q, 0.5), pd.sphere_centers(Q, MAX_WIDTH))


def test_tcp_centred_between_finger_pads(pd):
    """TCP lies midway between the pads, offset along hand z; the pad gap equals w."""
    rng = np.random.default_rng(12)
    for _ in range(10):
        q, w = pd.random_q(rng), rng.uniform(0, MAX_WIDTH)
        T = pd.link_transforms(q, w)
        z = T[8, :3, 2]
        mid = 0.5 * (T[9, :3, 3] + T[10, :3, 3])
        assert np.allclose(mid + (TCP_OFFSET - FINGER_Z) * z, pd.tcp(q)[:3, 3], atol=1e-12)
        assert np.isclose(np.linalg.norm(T[9, :3, 3] - T[10, :3, 3]), w, atol=1e-12)
        assert np.isclose((T[9, :3, 3] - T[10, :3, 3]) @ T[8, :3, 1], w)   # left finger on +y


def test_jacobian_point_batched_q(pd):
    rng = np.random.default_rng(13)
    Q, P = pd.random_q(rng, 5), rng.normal(scale=0.4, size=(5, 3))
    JB = pd.jacobian_point(Q, "panda_link6", P)
    assert JB.shape == (5, 3, 7)
    for n in range(5):
        assert np.allclose(JB[n], pd.jacobian_point(Q[n], "panda_link6", P[n]), atol=1e-14)


def test_jacobians_mutually_consistent(pd):
    """TCP and sphere Jacobians agree with jacobian_point; link-k spheres ignore joints after k."""
    rng = np.random.default_rng(14)
    for _ in range(5):
        q, w = pd.random_q(rng), rng.uniform(0, MAX_WIDTH)
        J = pd.jacobian_tcp(q)
        p_tcp = pd.tcp(q)[:3, 3]
        assert np.allclose(J[:3], pd.jacobian_point(q, "panda_hand", p_tcp), atol=1e-14)
        assert np.allclose(J[3:], pd.link_transforms(q)[1:8, :3, 2].T, atol=1e-14)
        c, _, Js, ids = pd.spheres(q, w)
        g = np.concatenate([pd.sphere_groups["hand"], pd.sphere_groups["fingers"]])
        assert np.allclose(Js[g], pd.jacobian_point(q, "panda_hand", c[g]), atol=1e-14)
        for s in range(len(ids)):
            nj = min(int(ids[s]), 7)
            assert np.all(Js[s][:, nj:] == 0.0)


def test_spheres_batched_q(pd):
    rng = np.random.default_rng(15)
    Q, W = pd.random_q(rng, 4), np.array([0.0, 0.02, 0.05, 0.08])
    c, r, J, ids = pd.spheres(Q, W)
    assert c.shape == (4, pd.n_spheres, 3) and J.shape == (4, pd.n_spheres, 3, 7)
    for n in range(4):
        cn, rn, Jn, idn = pd.spheres(Q[n], W[n])
        assert np.allclose(c[n], cn, atol=1e-14) and np.allclose(J[n], Jn, atol=1e-14)
        assert np.array_equal(r, rn) and np.array_equal(ids, idn)


def test_sphere_model_read_only(pd):
    """In-place edits to the shared sphere arrays (e.g. r += margin) raise."""
    _, r, _, ids = pd.spheres(pd.q_home)
    r0 = r.copy()
    with pytest.raises(ValueError):
        r += 0.01
    with pytest.raises(ValueError):
        ids[0] = 3
    with pytest.raises(ValueError):
        pd.sphere_groups["arm"][0] = 0
    assert np.array_equal(pd.spheres(pd.q_home)[1], r0)


def test_spheres_without_model():
    p = Panda(sphere_file=None)
    c, r, J, ids = p.spheres(p.q_home)
    assert c.shape == (0, 3) and r.shape == (0,) and J.shape == (0, 3, 7) and ids.shape == (0,)
    assert p.n_spheres == 0 and p.sphere_centers(np.zeros((3, 7))).shape == (3, 0, 3)


def test_spheres_clear_of_table_at_home(pd):
    """The home pose starts well inside the table barrier's safe set."""
    for w in (0.0, MAX_WIDTH):
        c, r, _, _ = pd.spheres(pd.q_home, w)
        assert (c[:, 2] - r).min() > 0.1


def test_quaternion_convention_wxyz():
    for th in (0.3, -1.2, 2.9):
        q = matrix_to_quat(rotz(th))
        assert np.allclose(q, np.array([np.cos(th / 2), 0, 0, np.sin(th / 2)]) * np.sign(np.cos(th / 2)))
    assert np.allclose(matrix_to_quat(np.diag([1.0, -1.0, -1.0])), [0, 1, 0, 0])   # top-down hand
    Rs = Rotation.random(50, random_state=3)
    Q = matrix_to_quat(Rs.as_matrix())
    xyzw = Rs.as_quat()
    ref = np.concatenate([xyzw[:, 3:], xyzw[:, :3]], 1)
    ref *= np.where(ref[:, :1] < 0, -1.0, 1.0)
    assert Q.shape == (50, 4) and np.allclose(Q, ref, atol=1e-12)
    assert np.allclose(quat_to_matrix(Q), Rs.as_matrix(), atol=1e-12)


def test_pose_error_moves_current_onto_goal():
    """pose_error(T_cur, T_goal) is the world-frame correction that moves T_cur onto T_goal."""
    rng = np.random.default_rng(16)
    angles = list(rng.uniform(0, np.pi, 30)) + [np.pi - 1e-9, np.pi - 1e-4, np.pi]
    for ang in angles:
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        Tc, Tg = np.eye(4), np.eye(4)
        Tc[:3, :3], Tc[:3, 3] = Rotation.random(random_state=int(rng.integers(1e6))).as_matrix(), rng.normal(size=3)
        Tg[:3, :3] = Rotation.from_rotvec(ang * a).as_matrix() @ Tc[:3, :3]
        Tg[:3, 3] = rng.normal(size=3)
        e = pose_error(Tc, Tg)
        assert np.allclose(Tc[:3, 3] + e[:3], Tg[:3, 3])
        assert np.allclose(Rotation.from_rotvec(e[3:]).as_matrix() @ Tc[:3, :3], Tg[:3, :3], atol=1e-7)
        assert np.linalg.norm(e[3:]) <= np.pi + 1e-12
        assert np.allclose(rot_log(np.eye(3)), 0.0)


def test_ik_multi_honours_q_ref(pd):
    """With q_ref set, ik_multi stays on q_ref's IK branch."""
    rng = np.random.default_rng(17)
    n_ok, far = 0, 0
    for _ in range(100):
        qr = pd.random_q(rng, margin=0.3)
        T = pd.tcp(qr + rng.normal(scale=0.02, size=7))
        q, ok = pd.ik_multi(T, q_ref=qr)
        n_ok += ok
        far += ok and np.abs(q - qr).max() > 0.2
    assert n_ok >= 98 and far == 0


def test_ik_multi_seed_list_edge_cases(pd):
    T = pd.tcp(pd.q_home)
    for kw in (dict(seeds=[]), dict(seeds=np.zeros((0, 7))),
               dict(seeds=pd.q_home), dict(seeds=[pd.q_home, pd.q_home + 0.1])):
        q, ok = pd.ik_multi(T, **kw)
        assert ok and pd.in_limits(q), kw
    # random seeds only: no success guarantee, just no crash
    q, ok = pd.ik_multi(T, seeds=[], n_random=3, rng=np.random.default_rng(0))
    assert q.shape == (7,) and pd.in_limits(q)
    # here q_ref only picks among solutions; it is not added as a seed
    q, ok = pd.ik_multi(T, seeds=[pd.q_home], q_ref=np.zeros(7), return_first=False)
    assert ok


def test_ik_nonfinite_inputs(pd):
    T = pd.tcp(pd.q_home)
    seed = pd.q_home.copy()
    seed[[2, 5]] = [np.nan, np.inf]
    q, ok = pd.ik(T, seed)
    assert ok and np.all(np.isfinite(q)) and pd.in_limits(q)
    q, ok = pd.ik_multi(T, seeds=[seed])
    assert ok and np.all(np.isfinite(q))
    Tn = T.copy()
    Tn[1, 3] = np.nan
    t = time.perf_counter()
    q, ok, info = pd.ik(Tn, pd.q_home, return_info=True)
    assert not ok and info["iters"] == 0 and np.all(np.isfinite(q)) and pd.in_limits(q)
    q, ok = pd.ik_multi(Tn, flip_z=True, n_random=5)
    assert not ok and np.all(np.isfinite(q)) and pd.in_limits(q)
    assert time.perf_counter() - t < 0.5


def test_ik_continuity_small_displacement(pd):
    """Seeded near the solution, IK does not jump to another branch."""
    rng = np.random.default_rng(18)
    worst = 0.0
    for _ in range(100):
        q0 = pd.random_q(rng, margin=0.2)
        dq = rng.normal(scale=0.01, size=7)
        q, ok = pd.ik(pd.tcp(q0 + dq), q0)
        assert ok
        worst = max(worst, np.abs(q - q0).max() / np.abs(dq).max())
    assert worst < 3.0


def test_ik_ok_flag_is_honest_and_limits_hold(pd):
    """q always stays in limits, and ok=True means the given tolerances are met."""
    rng = np.random.default_rng(19)
    for n in range(60):
        T = pd.tcp(pd.random_q(rng))
        if n % 3 == 1:
            T[:3, 3] *= 1.6                                         # mostly unreachable
        seed = pd.q_home + rng.normal(scale=3.0, size=7)            # often outside the limits
        pos_tol, rot_tol = (5e-4, 5e-3) if n % 2 else (2e-3, 2e-2)
        q, ok = pd.ik(T, seed, pos_tol=pos_tol, rot_tol=rot_tol)
        assert pd.in_limits(q, tol=0.0)
        if ok:
            e = pose_error(pd.tcp(q), T)
            assert np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < rot_tol


def test_ik_targets_with_joints_on_limits(pd):
    rng = np.random.default_rng(20)
    solved = 0
    for _ in range(40):
        q0 = pd.random_q(rng, margin=0.3)
        j = rng.choice(7, 2, replace=False)
        q0[j] = np.where(rng.random(2) < 0.5, pd.q_lower[j], pd.q_upper[j])
        q, ok = pd.ik(pd.tcp(q0), np.clip(q0 + rng.normal(scale=0.05, size=7), pd.q_lower, pd.q_upper))
        assert pd.in_limits(q, tol=0.0)
        solved += ok
    assert solved >= 36


def test_default_seeds_within_limits(pd):
    S = pd.default_ik_seeds()
    assert S.shape[1] == 7 and all(pd.in_limits(s) for s in S)
    assert np.allclose(S[0], pd.q_home)


def test_jacobian_tcp_speed(pd):
    q = pd.q_home
    best = np.inf
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(500):
            pd.jacobian_tcp(q)
        best = min(best, (time.perf_counter() - t) / 500)
    assert best < 1.5e-4
