"""Unit tests for safegrasp.grasping.safety_rank."""
import json
import time

import numpy as np
import pytest

from safegrasp.grasping import antipodal as ant
from safegrasp.grasping import safety_rank as sr
from safegrasp.grasping.types import Grasp, frame_from_axes, make_pose, pregrasp_pose, to_dict
from safegrasp.kinematics.panda import Panda
from safegrasp.perception.voxel import DistanceField, VoxelGrid

from test_grasping import box_surface, cylinder_surface, split

TARGET_XY = (0.55, 0.0)
TARGET_SIZE = (0.05, 0.05, 0.08)
GAP = 0.05                                        # target face -> neighbour face (+y side)
NB_LO = np.array([0.50, 0.025 + GAP, 0.0])        # tall neighbour, 10 x 5 x 20 cm
NB_HI = np.array([0.60, 0.025 + GAP + 0.05, 0.20])
BLOB_LO = np.array([0.44, -0.04, 0.0])            # unknown blob on the -x side of the target
BLOB_HI = np.array([0.51, 0.04, 0.14])
SAFE_KEYS = ("ik_ok", "geom_score", "joint_dist", "flipped", "h_min_obs_approach", "h_min_obs_transit",
             "h_min_obs", "h_min_unk", "unk_violation_frac", "h_min_table_approach", "h_min_table",
             "approach_dev", "q_pregrasp", "q_grasp")


@pytest.fixture(scope="module")
def panda():
    return Panda()


@pytest.fixture(scope="module")
def grid():
    return VoxelGrid()


def box_mask(grid, lo, hi):
    C = grid.centers()
    return np.all((C >= lo) & (C <= hi), axis=-1)


def field_from_boxes(grid, *boxes):
    m = np.zeros(grid.shape, bool)
    for lo, hi in boxes:
        m |= box_mask(grid, lo, hi)
    return DistanceField.from_mask(m, grid)


def grasp(pos, closing, width, score, approach=(0.0, 0.0, -1.0), tag=0.0):
    R = frame_from_axes(np.asarray(closing, float), np.asarray(approach, float))
    return Grasp(make_pose(R, pos), width, score, "antipodal", {"tag": float(tag)})


def top_grasp(yaw, score, width=0.07, z=0.06, tilt=0.0, tilt_dir=(1.0, 0.0, 0.0), tag=0.0):
    """Grasp on the target closing at yaw, approach tilted toward tilt_dir."""
    a = np.array([0.0, 0.0, -1.0]) + tilt * np.asarray(tilt_dir, float)
    return grasp([TARGET_XY[0], TARGET_XY[1], z], [np.cos(yaw), np.sin(yaw), 0.0], width, score, a, tag)


def check_output(out, panda, cfg=None):
    """Invariants of every rank_safe output."""
    cfg = sr.SafetyRankConfig() if cfg is None else cfg
    scores = [g.score for g in out]
    assert scores == sorted(scores, reverse=True)
    for g in out:
        assert g.method == "safe"
        for k in SAFE_KEYS:
            assert k in g.info
        json.dumps(to_dict(g))
        q_pre, q_g = np.array(g.info["q_pregrasp"]), np.array(g.info["q_grasp"])
        assert panda.in_limits(q_pre) and panda.in_limits(q_g)
        # IK solutions reach the returned (possibly flipped) poses
        for q, T in ((q_g, g.T), (q_pre, pregrasp_pose(g.T, cfg.pregrasp_dist))):
            Tq = panda.tcp(q)
            assert np.linalg.norm(Tq[:3, 3] - T[:3, 3]) < cfg.ik_pos_tol
            assert np.arccos(np.clip((np.trace(Tq[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)) < cfg.ik_rot_tol
        assert np.max(np.abs(q_g - q_pre)) <= cfg.max_joint_jump
        assert g.info["approach_dev"] < 0.02


def test_plan_joint_path_and_corridor(panda):
    q0, q1 = panda.q_home, panda.q_home + 0.3
    P = sr.plan_joint_path(q0, q1, 10)
    assert P.shape == (10, 7) and np.allclose(P[0], q0) and np.allclose(P[-1], q1)
    assert np.allclose(np.diff(P, axis=0), (q1 - q0) / 9)
    cfg = sr.SafetyRankConfig(n_transit=6, n_approach=4)
    Q, is_app = sr.corridor(q0, q1, q1 + 0.1, cfg)
    assert Q.shape == (10, 7) and is_app.tolist() == [False] * 6 + [True] * 4
    assert np.allclose(Q[5], q1) and np.allclose(Q[6], q1) and np.allclose(Q[-1], q1 + 0.1)


def test_flip_and_seeds(panda):
    T = top_grasp(0.3, 0.5).T
    Tf = sr.flip_about_approach(T)
    assert np.allclose(Tf[:3, 2], T[:3, 2]) and np.allclose(Tf[:3, :2], -T[:3, :2])
    assert np.allclose(sr.flip_about_approach(Tf), T)
    S = sr.ik_seeds(panda, panda.q_home)
    assert np.allclose(S[0], panda.q_home) and len(S) == 3        # q_seed == q_home deduplicated
    assert np.all(S >= panda.q_lower) and np.all(S <= panda.q_upper)
    S = sr.ik_seeds(panda, panda.q_home + 0.2)
    assert np.allclose(S[0], panda.q_home + 0.2) and np.allclose(S[1], panda.q_home) and len(S) == 6


def test_wrist_flip_reaches_flipped_pose(panda):
    q = panda.q_home.copy()
    wf = sr._wrist_flip(panda, q, q)
    assert wf is not None and np.isclose(abs(wf[0][6] - q[6]), np.pi)
    assert np.allclose(panda.tcp(wf[0]), sr.flip_about_approach(panda.tcp(q)), atol=1e-9)


def test_neighbour_closing_axis(panda, grid):
    """With a tall neighbour on +y, x-closing beats the higher-scored y-closing grasp."""
    obs = field_from_boxes(grid, (NB_LO, NB_HI))
    gs = [top_grasp(np.pi / 2, 0.9, tag=1), top_grasp(0.0, 0.7, tag=2), top_grasp(np.pi / 4, 0.6, tag=3)]
    out = sr.rank_safe(gs, panda, {"obs": (obs, 0.01)}, panda.q_home)
    check_output(out, panda)
    assert len(out) == 3
    top = out[0]
    assert top.info["tag"] == 2
    assert abs(top.closing[1]) < 0.3
    assert top.info["h_min_obs"] > 0 and top.info["h_min_obs_approach"] > 0
    bad = next(g for g in out if g.info["tag"] == 1)
    assert bad.info["h_min_obs_approach"] < 0              # blocked corridor -> penalty
    assert bad.score < bad.info["geom_score"] - 1.0
    # without 'unk' field: no unknown-space terms
    assert all(g.info["h_min_unk"] == sr.H_FAR and g.info["unk_violation_frac"] == 0.0 for g in out)


def test_neighbour_antipodal_end_to_end(panda, grid):
    """On real proposer output the best grasp keeps clearance."""
    tgt = box_surface(TARGET_XY, TARGET_SIZE)
    nb = box_surface((0.55, 0.025 + GAP + 0.025), (0.10, 0.05, 0.20))
    cyl = cylinder_surface((0.45, -0.12), 0.03, 0.12)
    tp, tn, op = split([tgt, nb, cyl])
    C = grid.centers()
    cyl_mask = (np.linalg.norm(C[..., :2] - np.array([0.45, -0.12]), axis=-1) <= 0.03) & (C[..., 2] <= 0.12)
    obs = DistanceField.from_mask(box_mask(grid, NB_LO, NB_HI) | cyl_mask, grid)
    gs = ant.propose_antipodal(tp, tn, op, np.random.default_rng(0), None)
    assert len(gs) >= 24
    out = sr.rank_safe(gs, panda, {"obs": (obs, 0.01)}, panda.q_home)
    check_output(out, panda)
    assert 0 < len(out) <= 24
    top = out[0]
    assert top.info["h_min_obs"] > 0
    assert abs(top.closing[1]) < 0.5
    blocked = [g for g in out if g.info["h_min_obs_approach"] < 0]
    assert all(g.score < top.score - 1.0 for g in blocked)


def test_unknown_blob_avoided(panda, grid):
    obs = DistanceField.empty_field(grid)
    unk = field_from_boxes(grid, (BLOB_LO, BLOB_HI))
    gs = [top_grasp(0.0, 0.9, tag=1),                                   # finger through the blob
          top_grasp(np.pi / 2, 0.7, tag=2),
          top_grasp(np.pi / 2, 0.8, tilt=0.6, tilt_dir=(1, 0, 0), tag=3),    # hand over the blob
          top_grasp(np.pi / 2, 0.75, tilt=0.6, tilt_dir=(-1, 0, 0), tag=4)]  # hand on the far side
    fields = {"obs": (obs, 0.01), "unk": (unk, 0.005)}
    out = sr.rank_safe(gs, panda, fields, panda.q_home)
    check_output(out, panda)
    by_tag = {g.info["tag"]: g for g in out}
    assert out[0].info["tag"] in (2, 4)
    assert out[0].info["h_min_unk"] > 0 and out[0].info["unk_violation_frac"] == 0
    for t in (1, 3):
        assert by_tag[t].info["h_min_unk"] < 0 and by_tag[t].info["unk_violation_frac"] > 0
    assert by_tag[4].score > by_tag[3].score
    # obstacle clearance is the same for all
    assert len({round(g.info["h_min_obs"], 9) for g in out}) == 1


def test_finger_grace_in_target_shadow(panda, grid):
    """The target's own shadow is ignored near the grasp, unless finger_grace = 0."""
    unk = field_from_boxes(grid, ([0.515, -0.07, 0.0], [0.585, 0.07, 0.07]))
    obs = DistanceField.empty_field(grid)
    gs = [top_grasp(np.pi / 2, 0.7)]
    fields = {"obs": (obs, 0.01), "unk": (unk, 0.005)}
    out = sr.rank_safe(gs, panda, fields, panda.q_home)
    assert out[0].info["h_min_unk"] > 0 and out[0].info["unk_violation_frac"] == 0
    out0 = sr.rank_safe(gs, panda, fields, panda.q_home, sr.SafetyRankConfig(finger_grace=0.0))
    assert out0[0].info["h_min_unk"] < 0 and out0[0].info["unk_violation_frac"] > 0
    assert out0[0].score < out[0].score
    # 'obs' has no grace zone
    out_obs = sr.rank_safe(gs, panda, {"obs": (unk, 0.01)}, panda.q_home)
    assert out_obs[0].info["h_min_obs_approach"] < 0


def test_table_barrier_penalised(panda, grid):
    """A low side grasp hits the table and ranks last."""
    obs = DistanceField.empty_field(grid)
    g_side = grasp([0.55, 0.0, 0.03], (0, 0, 1), 0.06, 0.9, approach=(1, 0, -0.15), tag=1)
    g_top = top_grasp(np.pi / 2, 0.5, tag=2)
    out = sr.rank_safe([g_side, g_top], panda, {"obs": (obs, 0.01)}, panda.q_home)
    assert [g.info["tag"] for g in out] == [2, 1]
    assert out[1].info["h_min_table_approach"] < 0 and out[0].info["h_min_table"] > 0
    off = sr.rank_safe([g_side], panda, {"obs": (obs, 0.01)}, panda.q_home, {"table_margin": None})
    assert off[0].info["h_min_table"] == sr.H_FAR and off[0].score > out[1].score


def test_far_grasp_dropped(panda, grid):
    obs = field_from_boxes(grid, (NB_LO, NB_HI))
    far = grasp([2.0, 0.0, 0.1], (0, 1, 0), 0.05, 1.0, tag=9)
    ok = top_grasp(0.0, 0.5, tag=1)
    out = sr.rank_safe([far, ok], panda, {"obs": (obs, 0.01)}, panda.q_home)
    assert [g.info["tag"] for g in out] == [1]
    assert sr.rank_safe([far], panda, {"obs": (obs, 0.01)}, panda.q_home) == []
    # a bare field (margin 0) is accepted too
    out0 = sr.rank_safe([far, ok], panda, {"obs": obs}, panda.q_home)
    assert np.isclose(out0[0].info["h_min_obs"], out[0].info["h_min_obs"] + 0.01)
    assert sr.rank_safe([], panda, {"obs": (obs, 0.01)}, panda.q_home) == []
    assert sr.solve_grasp_ik(panda, far, panda.q_home) is None


def test_flip_prefers_short_wrist_motion(panda, grid):
    """A grasp with hand x toward the base is flipped to cut joint motion."""
    g = grasp([0.5, 0.05, 0.05], (0, 1, 0), 0.05, 0.6)    # closing +y -> hand x = y x a = -world x
    assert g.T[0, 0] < -0.99
    var = sr.grasp_ik_variants(panda, g, panda.q_home)
    assert len(var) == 2 and [v.flipped for v in var] == [False, True]
    assert var[1].joint_dist < var[0].joint_dist
    obs = DistanceField.empty_field(grid)
    out = sr.rank_safe([g], panda, {"obs": (obs, 0.01)}, panda.q_home)
    check_output(out, panda)
    assert out[0].info["flipped"] == 1.0 and np.allclose(out[0].T, sr.flip_about_approach(g.T))
    best = sr.solve_grasp_ik(panda, g, panda.q_home)
    assert best.flipped and np.allclose(best.q_grasp, out[0].info["q_grasp"])
    assert np.isclose(best.joint_dist, out[0].info["joint_dist"])
    out_nf = sr.rank_safe([g], panda, {"obs": (obs, 0.01)}, panda.q_home, {"try_flip": False})
    assert out_nf[0].info["flipped"] == 0.0 and np.allclose(out_nf[0].T, g.T)


def test_solve_grasp_ik_continuity(panda):
    g = top_grasp(0.4, 0.5, tilt=0.4, tilt_dir=(-1, 0.3, 0))
    sol = sr.solve_grasp_ik(panda, g, panda.q_home)
    assert sol is not None and sol.grasp is not g
    assert np.allclose(panda.tcp(sol.q_grasp)[:3, 3], sol.grasp.T[:3, 3], atol=2e-3)
    assert np.allclose(panda.tcp(sol.q_pre)[:3, 3], pregrasp_pose(sol.grasp.T, 0.12)[:3, 3], atol=2e-3)
    assert np.max(np.abs(sol.q_grasp - sol.q_pre)) < 0.8
    assert np.isclose(sol.joint_dist, np.linalg.norm(sol.q_pre - panda.q_home)
                      + np.linalg.norm(sol.q_grasp - sol.q_pre))
    # the joint-space approach stays near the straight TCP line
    Q = sr.plan_joint_path(sol.q_pre, sol.q_grasp, 10)
    P = panda.tcp(Q)[:, :3, 3]
    a, b = P[0], P[-1]
    t = np.clip((P - a) @ (b - a) / ((b - a) @ (b - a)), 0, 1)
    assert np.max(np.linalg.norm(P - (a + t[:, None] * (b - a)), axis=1)) < 0.01


def test_score_formula_and_top_n(panda, grid):
    obs = field_from_boxes(grid, (NB_LO, NB_HI))
    unk = field_from_boxes(grid, (BLOB_LO, BLOB_HI))
    rng = np.random.default_rng(3)
    gs = [top_grasp(rng.uniform(0, np.pi), rng.uniform(0.2, 0.9), tilt=rng.uniform(0, 0.5),
                    tilt_dir=(np.cos(t), np.sin(t), 0), tag=i)
          for i, t in enumerate(rng.uniform(-np.pi, np.pi, 12))]
    cfg = sr.SafetyRankConfig(top_n=5, w_clear=0.4, w_unk=0.2, w_joint=0.3)
    out = sr.rank_safe(gs, panda, {"obs": (obs, 0.01), "unk": (unk, 0.005)}, panda.q_home, cfg)
    check_output(out, panda, cfg)
    top5 = {g.info["tag"] for g in sorted(gs, key=lambda g: -g.score)[:5]}
    assert 0 < len(out) <= 5 and {g.info["tag"] for g in out} <= top5
    c = cfg.h_clip
    for g in out:
        i = g.info
        assert np.isclose(i["h_min_obs"], min(i["h_min_obs_approach"], i["h_min_obs_transit"]))
        blocked = i["h_min_obs_approach"] < 0 or i["h_min_table_approach"] < 0
        s = (i["geom_score"] + cfg.w_clear * np.clip(i["h_min_obs"], -c, c) / c
             + cfg.w_unk * np.clip(i["h_min_unk"], -c, c) / c - cfg.w_joint * i["joint_dist"] / np.pi
             - cfg.penalty * blocked - cfg.unk_penalty * (i["h_min_unk"] < 0))
        assert np.isclose(g.score, s)
        assert i["geom_score"] == next(h.score for h in gs if h.info["tag"] == i["tag"])


def test_speed_24_candidates(panda, grid):
    tgt = box_surface(TARGET_XY, TARGET_SIZE)
    nb = box_surface((0.55, 0.025 + GAP + 0.025), (0.10, 0.05, 0.20))
    tp, tn, op = split([tgt, nb])
    gs = ant.propose_antipodal(tp, tn, op, np.random.default_rng(1), None)
    assert len(gs) >= 24
    fields = {"obs": (field_from_boxes(grid, (NB_LO, NB_HI)), 0.01),
              "unk": (field_from_boxes(grid, (BLOB_LO, BLOB_HI)), 0.005)}
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = sr.rank_safe(gs, panda, fields, panda.q_home)
        ts.append(time.perf_counter() - t0)
    assert len(out) > 0
    assert min(ts) < 0.5, ts


def test_straighten_approach(panda, grid):
    """Straightened approach samples follow the straight TCP line."""
    rng = np.random.default_rng(5)
    gs = []
    for i in range(12):
        pos = [rng.uniform(0.4, 0.7), rng.uniform(-0.25, 0.25), rng.uniform(0.02, 0.1)]
        a = [rng.uniform(-0.9, 0.9), rng.uniform(-0.9, 0.9), -1.0]
        gs.append(grasp(pos, np.cross(a, rng.normal(size=3)), 0.05, rng.random(), approach=a, tag=i))
    obs = DistanceField.empty_field(grid)
    raw = {g.info["tag"]: g for g in sr.rank_safe(gs, panda, {"obs": (obs, 0.01)}, panda.q_home,
                                                  {"approach_iters": 0, "top_n": 12})}
    out = sr.rank_safe(gs, panda, {"obs": (obs, 0.01)}, panda.q_home, {"top_n": 12})
    check_output(out, panda)
    assert len(out) == len(raw) >= 8
    dev0 = np.array([raw[g.info["tag"]].info["approach_dev"] for g in out])
    dev = np.array([g.info["approach_dev"] for g in out])
    assert dev0.max() > 0.005 and np.all(dev < 0.002) and np.all(dev <= dev0 + 1e-9)
    # direct call: end samples untouched, interior samples on the line
    g = out[0]
    Q, is_app = sr.corridor(panda.q_home, g.info["q_pregrasp"], g.info["q_grasp"])
    Qs = sr.straighten_approach(panda, Q[None], is_app, g.T[None], 0.12, 2)[0]
    idx = np.flatnonzero(is_app)
    assert np.allclose(Qs[~is_app], Q[~is_app]) and np.allclose(Qs[idx[[0, -1]]], Q[idx[[0, -1]]])
    P = panda.tcp(Qs[idx])
    line = pregrasp_pose(g.T, 0.12)[:3, 3] + np.linspace(0, 1, len(idx))[:, None] * 0.12 * g.T[:3, 2]
    assert np.max(np.linalg.norm(P[:, :3, 3] - line, axis=1)) < 0.002
    assert np.max(np.abs(P[:, :3, :3] - g.T[:3, :3])) < 0.02
