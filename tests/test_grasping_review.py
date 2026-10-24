"""Edge-case and regression tests for safegrasp.grasping."""
import inspect
import json

import numpy as np
import pytest
from scipy.spatial import cKDTree

from safegrasp.grasping import antipodal as ant
from safegrasp.grasping import gripper as grp
from safegrasp.grasping import safety_rank as sr
from safegrasp.grasping import topdown as td
from safegrasp.grasping.types import (DEFAULT_PREGRASP_DIST, TCP_OFFSET, TOPDOWN_QUAT, Grasp, R_from_quat,
                                      canonicalize_about_approach, frame_from_axes, hand_from_tcp, make_pose,
                                      pregrasp_pose, quat_from_R, to_dict)
from safegrasp.kinematics.panda import Panda
from safegrasp.perception.voxel import DistanceField, VoxelGrid

from test_grasping import box_surface, check_valid, cylinder_surface, split
from test_safety_rank import BLOB_HI, BLOB_LO, NB_HI, NB_LO, field_from_boxes, top_grasp

GEOM = grp.get_geometry()
BITS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=bool)


@pytest.fixture(scope="module")
def panda():
    return Panda()


def inflated_boxes(width, margin, sweep, geom=GEOM, include_wrist=True):
    lo, hi = geom.boxes(width, include_wrist)
    lo, hi = lo - margin, hi + margin
    lo[..., 2] -= sweep
    return lo, hi


def brute_collides(Ts, ws, pts, margin, sweep, geom=GEOM, include_wrist=True):
    """Brute-force reference: every point against every inflated, swept box."""
    out = []
    for T, w in zip(Ts, ws):
        lo, hi = inflated_boxes(w, margin, sweep, geom, include_wrist)
        q = (pts - T[:3, 3]) @ T[:3, :3]
        out.append(any(np.all((q >= l) & (q <= h), axis=1).any() for l, h in zip(lo, hi)))
    return np.array(out)


def rot_to(u, e):
    """Proper rotation R with R u = e (u, e unit)."""
    def basis(v):
        h = np.array([1.0, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1.0, 0])
        b = np.cross(v, h)
        b /= np.linalg.norm(b)
        return np.stack([v, b, np.cross(v, b)], axis=1)
    return basis(e) @ basis(u).T


def clutter_scene(seed):
    """Random target with 2-5 neighbours at least 5 mm away, seen from one camera."""
    rng = np.random.default_rng(seed)
    kind = rng.integers(3)
    txy = np.array([0.55, 0.0]) + rng.uniform(-0.05, 0.05, 2)
    if kind == 0:
        tgt = box_surface(txy, rng.uniform([0.025, 0.02, 0.04], [0.06, 0.05, 0.12]), yaw=rng.uniform(0, np.pi))
    elif kind == 1:
        tgt = cylinder_surface(txy, rng.uniform(0.015, 0.035), rng.uniform(0.05, 0.14))
    else:
        tgt = box_surface(txy, rng.uniform([0.03, 0.03, 0.02], [0.07, 0.06, 0.05]), yaw=rng.uniform(0, np.pi))
    clouds = [tgt]
    tree = cKDTree(tgt[0][:, :2])
    for _ in range(rng.integers(2, 6)):
        for _ in range(50):
            ang, d = rng.uniform(0, 2 * np.pi), rng.uniform(0.045, 0.11)
            c = txy + d * np.array([np.cos(ang), np.sin(ang)])
            nb = box_surface(c, rng.uniform([0.02, 0.02, 0.03], [0.06, 0.06, 0.2]), yaw=rng.uniform(0, np.pi))
            if tree.query(nb[0][:, :2])[0].min() > 0.005 and all(
                    cKDTree(o[0][:, :2]).query(nb[0][:, :2])[0].min() > 0.003 for o in clouds[1:]):
                clouds.append(nb)
                break
    return split(clouds)


def poses_widths(gs):
    return np.stack([g.T for g in gs]), np.array([g.width for g in gs])


def test_interface_signatures():
    names = lambda f: list(inspect.signature(f).parameters)[:5]
    assert names(td.propose_topdown)[:3] == ["target_pts", "rng", "cfg"]
    assert names(ant.propose_antipodal) == ["target_pts", "target_normals", "obstacle_pts", "rng", "cfg"]
    assert names(sr.rank_safe) == ["grasps", "panda", "fields", "q_seed", "cfg"]
    assert names(grp.collides)[:4] == ["T", "width", "pts", "margin"]
    assert names(grp.closing_region_points)[:3] == ["T", "width", "pts"]
    assert ant.AntipodalConfig().approach_dist == DEFAULT_PREGRASP_DIST == 0.12


def test_conventions_against_panda_model(panda):
    """TOPDOWN_QUAT, TCP offset and finger axis match the Panda kinematic model."""
    q = panda.q_home
    T_tcp = panda.tcp(q)
    # ready pose: pointing down, hand x = world x
    assert np.allclose(T_tcp[:3, :3], R_from_quat(TOPDOWN_QUAT), atol=1e-5)   # q7_home = 0.7854 ~ pi/4
    assert np.allclose(np.abs(quat_from_R(T_tcp[:3, :3])), TOPDOWN_QUAT, atol=1e-5)
    fk = panda.fk_all(q, width=0.06)
    assert np.allclose(hand_from_tcp(T_tcp), fk["panda_hand"], atol=1e-12)
    assert np.isclose(np.linalg.norm(T_tcp[:3, 3] - fk["panda_hand"][:3, 3]), TCP_OFFSET)
    # fingers sit at +-w/2 along TCP y
    names = [n for n in fk if "finger" in n]
    ys = sorted(float(((fk[n][:3, 3] - T_tcp[:3, 3]) @ T_tcp[:3, :3])[1]) for n in names)
    assert np.allclose(ys, [-0.03, 0.03], atol=1e-9)
    # gripper model: inner finger faces at about +-w/2
    lo, hi = GEOM.finger_boxes(0.06)
    assert abs(lo[0, 1] - 0.03) < 1e-3 and abs(hi[1, 1] + 0.03) < 1e-3


def test_quaternion_edge_cases():
    """Round trip of (w, x, y, z) quaternions, including w = 0 (rotations by pi)."""
    for q in ([0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0.6, 0.8, 0], [1e-9, 0, 0.6, 0.8]):
        q = np.asarray(q, float) / np.linalg.norm(q)
        q2 = quat_from_R(R_from_quat(q))
        assert np.allclose(q2, q, atol=1e-7) or np.allclose(q2, -q, atol=1e-7)
        assert np.allclose(R_from_quat(q2), R_from_quat(q), atol=1e-9)
    # Rz(90deg): (cos45, 0, 0, sin45) and maps x -> y
    q = quat_from_R(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]]))
    assert np.allclose(q, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])


def test_canonicalize_flip_invariant_including_ties(panda):
    flip = sr.flip_about_approach
    # radially closing top-down grasps: the two flips tie
    for p in ([0.5, 0.0, 0.05], [0.4, 0.3, 0.05], [0.45, -0.2, 0.1]):
        r = np.array(p[:2]) / np.linalg.norm(p[:2])
        T = make_pose(frame_from_axes([r[0], r[1], 0.0], [0, 0, -1.0]), p)
        c1, c2 = canonicalize_about_approach(T), canonicalize_about_approach(flip(T))
        assert np.allclose(c1, c2)
        assert np.allclose(canonicalize_about_approach(c1), c1)          # idempotent
    # the chosen flip keeps joint 7 far from its limits (q7 = q7_ready - pi/2)
    T = canonicalize_about_approach(make_pose(frame_from_axes([1.0, 0, 0], [0, 0, -1.0]), [0.5, 0.0, 0.05]))
    q, ok = panda.ik(T, panda.q_home)
    assert ok and min(q[6] - panda.q_lower[6], panda.q_upper[6] - q[6]) > 1.5
    rng = np.random.default_rng(0)
    Ts = []
    for _ in range(500):
        a = rng.normal(size=3)
        a[2] = -abs(a[2])
        Ts.append(make_pose(frame_from_axes(rng.normal(size=3), a), [rng.uniform(0.3, 0.7), rng.uniform(-.3, .3), .05]))
    Ts = np.stack(Ts)
    assert np.allclose(canonicalize_about_approach(Ts), canonicalize_about_approach(flip(Ts)))
    T0 = make_pose(R_from_quat(TOPDOWN_QUAT), [0.5, 0.1, 0.05])
    assert np.allclose(canonicalize_about_approach(flip(T0)), T0)


def test_boxes_reach_exact():
    rng = np.random.default_rng(1)
    for _ in range(50):
        a, b = rng.uniform(-0.2, 0.2, (2, 4, 3))
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        corners = np.stack([np.where(BITS, h, l) for l, h in zip(lo, hi)])
        assert np.isclose(grp.boxes_reach(lo, hi), np.linalg.norm(corners, axis=-1).max())
        P = rng.uniform(lo[0], hi[0], size=(2000, 3))
        assert np.linalg.norm(P, axis=1).max() <= grp.boxes_reach(lo, hi) + 1e-12


@pytest.mark.parametrize("margin,sweep,width", [(0.005, 0.0, 0.08), (0.01, 0.0, 0.03), (0.004, 0.12, 0.08),
                                                (0.0, 0.0, 0.08), (0.005, 0.0, 0.1)])
def test_prefilter_keeps_inflated_corners(margin, sweep, width):
    """A point just inside the farthest inflated corner survives the prefilter in every direction."""
    lo, hi = inflated_boxes(width, margin, sweep)
    C = np.stack([np.where(BITS, h, l) for l, h in zip(lo, hi)]).reshape(-1, 3)
    far = C[np.argmax(np.linalg.norm(C, axis=1))]
    u = far / np.linalg.norm(far)
    for e in np.vstack([np.eye(3), -np.eye(3)]):
        R = rot_to(u, e)
        assert np.allclose(R @ u, e) and np.isclose(np.linalg.det(R), 1.0)
        T = make_pose(R, [0.5, 0.0, 0.3])
        p = (T[:3, 3] + R @ (far * (1 - 1e-7)))[None]
        assert brute_collides(T[None], [width], p, margin, sweep)[0]
        assert grp.collides(T, width, p, margin, sweep=sweep)
        assert grp.CloudCollider(p, 0.005).collides(T[None], width, margin, sweep=sweep)[0]


def test_collision_fast_paths_match_bruteforce():
    """Fast collision checks match brute force on points within 1e-6 of box boundaries."""
    rng = np.random.default_rng(2)
    G = 60
    Ts = np.stack([make_pose(R_from_quat(rng.normal(size=4)), rng.uniform([0.2, -0.5, 0.0], [1.0, 0.5, 0.5]))
                   for _ in range(G)])
    ws = rng.uniform(0.0, 0.1, size=G)
    for margin, sweep in ((0.0, 0.0), (0.005, 0.0), (0.004, 0.12)):
        pts = []
        for T, w in zip(Ts[:20], ws[:20]):                   # boundary points of the first 20 poses
            lo, hi = inflated_boxes(w, margin, sweep)
            for l, h in zip(lo, hi):
                q = rng.uniform(l, h, size=(30, 3))
                k = rng.integers(3, size=30)
                q[np.arange(30), k] = np.where(rng.random(30) < 0.5, l[k], h[k]) + rng.choice([-1e-6, 1e-6], 30)
                corner = np.where(BITS[rng.integers(8)], h, l) * (1 + rng.choice([-1e-6, 1e-6]))
                pts.append(np.vstack([q, corner]) @ T[:3, :3].T + T[:3, 3])
        pts = np.concatenate(pts)
        ref = brute_collides(Ts, ws, pts, margin, sweep)
        assert 0 < ref.sum() < G
        assert np.array_equal(grp.collides_batch(Ts, ws, pts, margin, sweep=sweep), ref)
        for voxel in (0.003, 0.01):
            assert np.array_equal(grp.CloudCollider(pts, voxel).collides(Ts, ws, margin, sweep=sweep), ref)


def test_closing_volume_and_voxel_labels():
    T = make_pose(R_from_quat(TOPDOWN_QUAT), [0.5, 0.0, 0.05])
    # between the finger roots (z = -0.03), between the pads (z = 0), outside
    q = np.array([[0.0, 0.01, -0.03], [0.0, 0.01, 0.0], [0.0, 0.05, 0.0]])
    p = grp.tcp_to_world(T, q)
    assert grp.closing_region_mask(T, 0.06, p).tolist() == [False, True, False]
    assert grp.closing_region_mask(T, 0.06, p, full_length=True).tolist() == [True, True, False]
    rng = np.random.default_rng(3)
    P = rng.uniform(0.3, 0.7, size=(3000, 3))
    lab = grp.voxel_labels(P, 0.01)
    idx, _, mem, start, cnt = grp.voxel_downsample(P, 0.01, return_groups=True)
    assert lab.max() + 1 == len(idx)
    assert np.array_equal(grp.voxel_labels(P[:10], 0.0), np.arange(10))    # no reduction
    for v in range(0, len(idx), 37):                     # same grouping as voxel_downsample
        assert np.unique(lab[mem[start[v]:start[v] + cnt[v]]]).size == 1
    # count = distinct labels inside the closing volume
    Ts = np.stack([make_pose(R_from_quat(rng.normal(size=4)), rng.uniform(0.4, 0.6, 3)) for _ in range(10)])
    n = grp.closing_volume_voxel_counts(Ts, 0.08, P, lab)
    ref = [np.unique(lab[grp.closing_region_mask(T_, 0.08, P, full_length=True)]).size for T_ in Ts]
    assert n.tolist() == ref and max(ref) > 0


@pytest.mark.parametrize("seed", [14, 19])
def test_antipodal_final_check_exact_without_verifier(seed):
    """With search_inflation=1, no output collides even without the verifier."""
    tp, tn, op = clutter_scene(seed)
    cfg = ant.AntipodalConfig(verify=False, search_inflation=1.0)
    gs = ant.propose_antipodal(tp, tn, op, np.random.default_rng(seed), cfg)
    assert len(gs) > 10
    Ts, ws = poses_widths(gs)
    allp = np.concatenate([tp, op])
    assert not grp.collides_batch(Ts, ws, allp, cfg.margin, sweep=cfg.approach_dist).any()


@pytest.mark.parametrize("seed", [1, 11, 13, 14])
def test_antipodal_no_obstacle_between_fingers(seed):
    """No output closes on a neighbour."""
    tp, tn, op = clutter_scene(seed)
    cfg = ant.AntipodalConfig()
    gs = ant.propose_antipodal(tp, tn, op, np.random.default_rng(seed), cfg)
    assert len(gs) > 0
    check_valid(gs, tp, op)
    Ts, ws = poses_widths(gs)
    n = grp.closing_volume_voxel_counts(Ts, ws, op, grp.voxel_labels(op, cfg.voxel_obstacle))
    assert n.max() <= cfg.max_obstacle_in_region


def test_antipodal_pregrasp_corridor_clear():
    """The full 0.12 m approach from the pre-grasp is collision free, not just the last 0.10 m."""
    tp, tn, op = split([box_surface((0.55, 0.0), (0.04, 0.03, 0.06), yaw=0.3)])
    gs = ant.propose_antipodal(tp, tn, op, np.random.default_rng(0), None)
    Ts, ws = poses_widths(gs)
    rng = np.random.default_rng(1)
    lo, hi = GEOM.boxes(0.08)
    cand = np.concatenate([grp.tcp_to_world(pregrasp_pose(g, 0.12), rng.uniform(l, h, size=(400, 3)))
                           for g in gs[:10] for l, h in zip(lo, hi)])
    hit = np.zeros(len(cand), dtype=bool)                  # inside any 0.10 m-swept hand of the clean grasps
    for T, w in zip(Ts, ws):
        l6, h6 = inflated_boxes(w, 0.006, 0.10)
        q = (cand - T[:3, 3]) @ T[:3, :3]
        hit |= np.any([np.all((q >= l) & (q <= h), axis=1) for l, h in zip(l6, h6)], axis=0)
    wall = cand[~hit & (cand[:, 2] > 0.01)]
    assert len(wall) > 0
    obst = np.concatenate([op, wall])
    gs2 = ant.propose_antipodal(tp, tn, obst, np.random.default_rng(0), None)
    assert len(gs2) > 10
    allp = np.concatenate([tp, obst])
    for g in gs2:
        assert not grp.collides(pregrasp_pose(g, DEFAULT_PREGRASP_DIST), g.width, allp, 0.004)
    Ts2, ws2 = poses_widths(gs2)
    assert not grp.collides_batch(Ts2, ws2, allp, 0.004, sweep=DEFAULT_PREGRASP_DIST).any()


def test_antipodal_info_y_shift_in_output_frame():
    """TCP - depth * approach - y_shift * closing lands on a target point, up to the refine shift."""
    tp, tn, op = split([cylinder_surface((0.55, 0.0), 0.028, 0.12)])
    cfg = ant.AntipodalConfig(max_grasps=200)
    gs = ant.propose_antipodal(tp, tn, op, np.random.default_rng(4), cfg)
    tree = cKDTree(tp)
    dmax = max(abs(x) for x in cfg.refine_shifts) + 1e-9
    big = 0
    for g in gs:
        s = g.position - g.info["depth"] * g.approach - g.info["y_shift"] * g.closing
        assert tree.query(s)[0] <= dmax
        big += abs(g.info["y_shift"]) > 0.015
    assert big > 5                                       # the check has teeth


def test_antipodal_degenerate_inputs():
    tp, tn, op = split([box_surface((0.55, 0.0), (0.05, 0.04, 0.07))])
    assert ant.propose_antipodal(np.zeros((0, 3)), np.zeros((0, 3)), None, 0, None) == []
    assert ant.propose_antipodal(np.full((50, 3), np.nan), np.ones((50, 3)), op, 0, None) == []
    # all normals invalid -> nothing to sample
    assert ant.propose_antipodal(tp, np.zeros_like(tp), op, 0, None) == []
    # 20 x 20 cm top, too wide for the gripper
    wide = split([box_surface((0.55, 0.0), (0.2, 0.2, 0.03))])
    for g in ant.propose_antipodal(wide[0], wide[1], wide[2], 0, None):
        assert g.width <= 0.08                           # only edge / rim grasps can exist
    # target sunk below the table: no grasp may enter the table
    sunk = tp - np.array([0, 0, 0.069])
    for g in ant.propose_antipodal(sunk, tn, None, 0, None):
        assert grp.lowest_z(g.T, g.width) >= 0.0
    gs, st = ant.propose_antipodal(tp, tn, op, 0, {"max_grasps": 0}, return_stats=True)
    assert gs == [] and st["n_out"] == 0
    # stats always carry n_out, also on early exits
    assert ant.propose_antipodal(tp[:3], tn[:3], None, 0, None, return_stats=True)[1]["n_out"] == 0


def test_antipodal_perfect_box_is_antipodal():
    """Fully seen box: full antipodal credit only inside the friction cone, less for diagonals."""
    P, N = box_surface((0.55, 0.0), (0.04, 0.06, 0.08), yaw=0.2)
    cfg = ant.AntipodalConfig(max_grasps=200)
    gs = ant.propose_antipodal(P, N, None, np.random.default_rng(0), cfg)
    check_valid(gs, P, np.zeros((0, 3)))
    assert all(g.info["antipodal"] > 0.95 and g.info["one_sided"] == 0 for g in gs[:5])
    faces = np.array([[np.cos(0.2), np.sin(0.2), 0], [-np.sin(0.2), np.cos(0.2), 0], [0, 0, 1.0]])
    ang = np.array([np.degrees(np.arccos(np.abs(faces @ g.closing).max())) for g in gs])
    full = np.array([g.info["antipodal"] >= 0.999 and g.info["one_sided"] == 0 for g in gs])
    assert full.sum() >= 5 and ang[full].max() <= np.degrees(np.arctan(cfg.friction)) + 2.0
    diag = np.array([abs(np.degrees(np.arccos(np.abs(faces[:2] @ g.closing).max())) - 45) < 5 for g in gs])
    two = np.array([g.info["one_sided"] == 0 for g in gs])
    if np.any(diag & two):
        assert max(g.info["antipodal"] for g, d in zip(gs, diag & two) if d) < 0.8


def test_topdown_degenerate_inputs():
    cfg = td.TopDownConfig()
    rng = np.random.default_rng(0)
    cases = {
        "identical": np.tile([0.5, 0.0, 0.04], (20, 1)),
        "collinear": np.stack([np.linspace(0.45, 0.55, 30), np.zeros(30), np.full(30, 0.03)], 1),
        "vertical_line": np.stack([np.full(30, 0.5), np.zeros(30), np.linspace(0, 0.1, 30)], 1),
        "nan_rows": np.vstack([rng.uniform([0.5, -0.02, 0], [0.54, 0.02, 0.05], (40, 3)), np.full((5, 3), np.nan)]),
        "below_table": rng.uniform([0.5, -0.02, -0.05], [0.54, 0.02, -0.01], (40, 3)),
    }
    for name, P in cases.items():
        gs = td.propose_topdown(P, None, cfg)
        assert len(gs) >= 1, name
        for g in gs:
            R = g.T[:3, :3]
            assert np.all(np.isfinite(g.T)), name
            assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0.999, name
            assert np.allclose(g.approach, [0, 0, -1]), name
            assert cfg.min_width - 1e-12 <= g.width <= cfg.max_width + 1e-12, name
            assert g.position[2] >= cfg.table_z + cfg.min_tcp_z - 1e-12, name
            json.dumps(to_dict(g))
    assert td.propose_topdown(np.zeros((2, 3)), None, None) == []


def test_rank_safe_missing_fields_and_info_keys(panda):
    """A missing field is accepted, 'unk_frac' is returned and info values are floats."""
    grid = VoxelGrid()
    obs = field_from_boxes(grid, (NB_LO, NB_HI))
    g = top_grasp(0.0, 0.5)
    ref = sr.rank_safe([g], panda, {"obs": (obs, 0.01)}, panda.q_home)
    for fields in ({"obs": (obs, 0.01), "unk": (None, 0.005)}, {"obs": (obs, 0.01), "unk": None}):
        out = sr.rank_safe([g], panda, fields, panda.q_home)
        assert len(out) == 1 and np.isclose(out[0].score, ref[0].score)
        assert out[0].info["h_min_unk"] == sr.H_FAR
    none = sr.rank_safe([g], panda, {"obs": (None, 0.01)}, panda.q_home)
    assert none[0].info["h_min_obs"] == sr.H_FAR == none[0].info["h_min_obs_approach"]
    info = ref[0].info
    for k in ("h_min_obs", "h_min_unk", "ik_ok", "unk_frac"):
        assert k in info
    assert info["unk_frac"] == info["unk_violation_frac"]
    assert isinstance(info["ik_ok"], float) and info["ik_ok"] == 1.0
    json.dumps(to_dict(ref[0]))


def test_evaluate_corridors_matches_bruteforce(panda):
    """Batched barrier evaluation matches a per-sample loop."""
    grid = VoxelGrid()
    obs = field_from_boxes(grid, (NB_LO, NB_HI))
    unk = field_from_boxes(grid, (BLOB_LO, BLOB_HI))
    cfg = sr.SafetyRankConfig()
    var = [v for a in np.linspace(0, np.pi, 4) for v in sr.grasp_ik_variants(panda, top_grasp(a, 0.5), panda.q_home)]
    Q, is_app = zip(*(sr.corridor(panda.q_home, v.q_pre, v.q_grasp, cfg) for v in var))
    Q = np.stack(Q)
    ev = sr.evaluate_corridors(panda, Q, [v.grasp.width for v in var], np.stack([v.grasp.T[:3, 3] for v in var]),
                               is_app[0], {"obs": (obs, 0.01), "unk": (unk, 0.005)}, cfg)
    exempt = np.concatenate([panda.sphere_groups["hand"], panda.sphere_groups["fingers"]])
    ah = np.concatenate([panda.sphere_groups["arm"], panda.sphere_groups["hand"]])
    for k, v in enumerate(var):
        ho, hu, ht = [], [], []
        for q in Q[k]:
            c, r, _, _ = panda.spheres(q, v.grasp.width)
            ho.append((obs.query(c)[0] - r - 0.01).min())
            h = unk.query(c)[0] - r - 0.005
            if np.linalg.norm(panda.tcp(q)[:3, 3] - v.grasp.T[:3, 3]) < cfg.finger_grace:
                h[exempt] = np.inf
            hu.append(h.min())
            ht.append((c[ah, 2] - r[ah] - cfg.table_margin).min())
        ho, hu, ht = np.minimum(ho, sr.H_FAR), np.minimum(hu, sr.H_FAR), np.minimum(ht, sr.H_FAR)
        assert np.isclose(ev["h_min_obs"][k], ho.min()) and np.isclose(ev["h_min_unk"][k], hu.min())
        assert np.isclose(ev["h_min_obs_approach"][k], ho[is_app[0]].min())
        assert np.isclose(ev["h_min_table"][k], ht.min())
        assert np.isclose(ev["unk_frac"][k], np.mean(hu < 0))


def test_rank_safe_flip_keeps_info_consistent(panda):
    """A flipped grasp also flips the sign of info['y_shift']."""
    grid = VoxelGrid()
    R = frame_from_axes(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]))     # hand x = -world x
    g = Grasp(make_pose(R, [0.5, 0.05, 0.05]), 0.05, 0.6, "antipodal", {"y_shift": 0.01})
    out = sr.rank_safe([g], panda, {"obs": (DistanceField.empty_field(grid), 0.01)}, panda.q_home)
    assert out[0].info["flipped"] == 1.0
    assert np.allclose(out[0].closing, -g.closing) and out[0].info["y_shift"] == -0.01
    assert g.info["y_shift"] == 0.01                                               # input untouched
