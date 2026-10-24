"""Unit tests for deadlock recovery in safegrasp.pipeline."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from safegrasp.grasping.types import Grasp
from safegrasp.kinematics.panda import Panda
from safegrasp.pipeline.methods import get_method
from safegrasp.pipeline.recovery import RecoveryState, approach_resume_time
from safegrasp.pipeline.trial import Monitor, Percept, TrialCfg, execute


@pytest.mark.parametrize("mode", ["reobserve", "relax"])
@pytest.mark.parametrize("phase", ["transit", "approach"])
def test_stall_in_transit_or_approach_triggers(mode, phase):
    assert RecoveryState(mode, "oacbf").should_recover("stall", phase)


@pytest.mark.parametrize("status", ["done", "timeout"])
def test_only_stall_triggers(status):
    assert not RecoveryState("reobserve", "oacbf").should_recover(status, "approach")


@pytest.mark.parametrize("phase", ["retreat", "close", "lift", "hold"])
def test_other_phases_never_trigger(phase):
    assert not RecoveryState("relax", "oacbf").should_recover("stall", phase)


@pytest.mark.parametrize("executor", ["nominal", "apf", "cbf"])
def test_only_oacbf_executor_triggers(executor):
    assert not RecoveryState("reobserve", executor).should_recover("stall", "approach")


def test_mode_none_never_triggers():
    assert not RecoveryState("none", "oacbf").should_recover("stall", "approach")


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        RecoveryState("retry", "oacbf")


def test_cap_is_per_trial_across_phases():
    rec = RecoveryState("reobserve", "oacbf", max_recoveries=2)
    rec.record("transit", 1.0)
    assert rec.should_recover("stall", "approach")
    rec.record("approach", 4.0)
    assert rec.n == 2
    assert not rec.should_recover("stall", "transit")
    assert not rec.should_recover("stall", "approach")
    assert [e["k"] for e in rec.events] == [1, 2]
    assert [e["phase"] for e in rec.events] == ["transit", "approach"]


def test_cap_zero_disables_recovery():
    assert not RecoveryState("relax", "oacbf", max_recoveries=0).should_recover("stall", "approach")


def test_loop_pattern_stops_at_cap():
    """execute()'s retry loop stops at the cap."""
    rec = RecoveryState("relax", "oacbf", max_recoveries=2)
    st, runs = "stall", 1
    while rec.should_recover(st, "approach"):
        rec.record("approach", float(runs))
        st, runs = "stall", runs + 1
    assert (rec.n, runs, st) == (2, 3, "stall")


def test_resume_time_matches_depth():
    a, pg = np.array([0.0, 0.0, -1.0]), np.array([0.5, 0.0, 0.1])
    L, T = 0.12, 2.0
    assert approach_resume_time(pg - a * L, pg, a, L, T) == pytest.approx(0.0)
    assert approach_resume_time(pg, pg, a, L, T) == pytest.approx(T)
    # 4 cm past the pre-grasp, 3 cm off-axis: only the depth counts
    p = pg - a * 0.08 + np.array([0.03, 0.0, 0.0])
    assert approach_resume_time(p, pg, a, L, T) == pytest.approx(2.0 / 3.0)


def test_resume_time_clipped():
    a, pg = np.array([0.0, 0.0, -1.0]), np.array([0.5, 0.0, 0.1])
    assert approach_resume_time(pg - a * 0.2, pg, a, 0.12, 2.0) == 0.0   # behind the pre-grasp
    assert approach_resume_time(pg + a * 0.05, pg, a, 0.12, 2.0) == 2.0  # past the grasp


def test_method_spec_parses_recover():
    m = get_method("safe_oacbf@recover=reobserve")
    base = get_method("safe_oacbf")
    assert (m.name, m.recover, base.recover) == ("safe_oacbf@recover=reobserve", "reobserve", "none")
    assert m.with_(name=base.name, recover="none") == base  # nothing else changes


def test_method_spec_rejects_unknown_recover():
    with pytest.raises(ValueError):
        get_method("safe_oacbf@recover=bogus")


class Box:
    """Signed distance to an axis-aligned box (centre c, half extents h)."""

    def __init__(self, c, h):
        self.c, self.h = np.asarray(c, float), np.asarray(h, float)

    def _d(self, p):
        q = np.abs(p - self.c) - self.h
        return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)

    def query(self, p):
        p = np.atleast_2d(p)
        e = 1e-5
        g = np.stack([(self._d(p + e * u) - self._d(p - e * u)) / (2 * e) for u in np.eye(3)], axis=1)
        return self._d(p), g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)


class FarField:
    def query(self, p):
        p = np.atleast_2d(p)
        return np.full(len(p), 10.0), np.tile([0.0, 0.0, 1.0], (len(p), 1))


class FakeScene:
    """Kinematic fake: joints track the command, two static objects, no contacts."""

    def __init__(self, q0):
        self.q, self.w = np.array(q0, float), 0.08
        self.active = np.array([True, True])
        self.cfg = SimpleNamespace(dt=1.0 / 120.0, decimation=2)

    def robot_q(self):
        return self.q.copy()

    def control_step(self, q_cmd, width, qd):
        self.q, self.w = np.array(q_cmd, float), float(width)

    def gripper_width(self):
        return self.w

    def object_poses(self):
        return np.array([[0.5, 0.0, 0.02, 1, 0, 0, 0], [0.5, 0.3, 0.02, 1, 0, 0, 0]], float)

    def contact_matrix(self):
        return np.zeros((1, 2))

    def object_points_world(self, ids, p0):
        return []

    def body_pose(self, name):
        return np.zeros(7)


def _percept(f_unk, tag):
    return Percept(views=[], grid=None, labels=None, inst=None, target_pts=np.zeros((0, 3)),
                   target_normals=np.zeros((0, 3)), obstacle_pts=np.zeros((0, 3)), f_obs=FarField(), f_unk=f_unk,
                   stats=dict(n_unk_barrier=tag, perceive_s=0.0))


def _run(recover):
    """Approach onto an unknown-space block 5 cm below the TCP; re-observing removes it."""
    panda = Panda()
    q0 = panda.q_home.copy()
    T_pre = panda.tcp(q0)
    a = T_pre[:3, 2]
    T_g = T_pre.copy()
    T_g[:3, 3] += a * 0.12
    q_g, ok = panda.ik(T_g, q0)
    assert ok
    scene = FakeScene(q0)
    cfg = TrialCfg()
    method = get_method("safe_oacbf" if recover == "none" else f"safe_oacbf@recover={recover}")
    top = T_pre[:3, 3] + a * 0.05          # a = -z at the home pose
    wall = _percept(Box(top - np.array([0.0, 0.0, 0.1]), [0.15, 0.15, 0.1]), 100)
    mon = Monitor(scene, 0, panda, cfg)
    cands = [(Grasp(T_g, 0.04, 1.0, "test"), q0, q_g)]
    return execute(scene, method, panda, wall, cands, 0, mon, cfg, reperceive=lambda: _percept(FarField(), 0))


def test_closed_loop_without_recovery_deadlocks():
    res = _run("none")
    assert res["attempts"] == ["approach_stall"]
    assert res["fail"] == "approach_stall" and res["recoveries"] == 0


def test_closed_loop_reobserve_unsticks_same_candidate():
    res = _run("reobserve")
    assert res["recoveries"] == 1
    ev = res["recovery_events"][0]
    assert (ev["phase"], ev["after"], ev["n_unk_barrier_before"], ev["n_unk_barrier_after"]) == ("approach", "done", 100, 0)
    assert 0.0 < ev["t_resume"] < 2.0               # resumed mid-approach, not from the pre-grasp
    assert res["attempts"] == ["approach_done"] and res["phase_reached"] == "hold"


def test_closed_loop_relax_capped_at_two():
    # relaxing 5 mm cannot clear a 7 cm block
    res = _run("relax")
    assert res["recoveries"] == 2
    assert [e["after"] for e in res["recovery_events"]] == ["stall", "stall"]
    assert res["recovery_events"][0]["margin_unk_before"] == pytest.approx(0.005)
    assert res["recovery_events"][1]["margin_unk_before"] == 0.0
    assert res["fail"] == "approach_stall"
