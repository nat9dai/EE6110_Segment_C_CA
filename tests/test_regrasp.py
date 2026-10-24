"""Tests for the empty-close recovery search ``recover_grasps`` (pure geometry, no Isaac)."""
import numpy as np
import pytest

from safegrasp.grasping import gripper as grp
from safegrasp.grasping.types import Grasp, frame_from_axes, make_pose
from safegrasp.pipeline.trial import recover_grasps

GEOM = grp.get_geometry()
GX = GEOM.closing_x_scale * float(np.max(np.abs(GEOM.finger_x)))   # pad half-extent along hand x


def _grasp(R, t, width=0.08):
    return Grasp(T=make_pose(R, np.asarray(t, float)), width=width, score=1.0, method="test")


def _topdown_R():
    return frame_from_axes(closing=np.array([0.0, 1.0, 0.0]), approach=np.array([0.0, 0.0, -1.0]))


def _blob(centre, n=300, spread=0.004, seed=0):
    return np.asarray(centre, float) + spread * np.random.default_rng(seed).standard_normal((n, 3))


def _brackets(g, pts):
    """Count the points of ``pts`` inside g's pad closing volume (TCP frame)."""
    R, t = g.T[:3, :3], g.T[:3, 3]
    loc = (np.asarray(pts) - t) @ R
    pz0, pz1 = GEOM.pad_z
    return int(((np.abs(loc[:, 0]) <= GX) & (np.abs(loc[:, 1]) < 0.5 * g.width)
                & (loc[:, 2] >= pz0) & (loc[:, 2] <= pz1)).sum())


def test_empty_cloud_returns_nothing():
    g = _grasp(_topdown_R(), [0.5, 0.0, 0.2])
    assert recover_grasps(g, np.zeros((0, 3))) == []


def test_unreachable_target_returns_nothing():
    """A target beyond max_shift gives no recovery."""
    g = _grasp(_topdown_R(), [0.5, 0.0, 0.15])
    pts = _blob([0.5 + 0.30, 0.0, 0.15], seed=1)
    assert recover_grasps(g, pts, max_shift=0.03) == []


def test_lateral_miss_is_recentred_and_brackets():
    """A grasp that just misses the object sideways is moved to bracket it."""
    R = _topdown_R()
    t = np.array([0.5, 0.0, 0.15])
    g = _grasp(R, t)
    centre = np.array([0.5 + GX + 0.012, 0.0, 0.15])
    pts = _blob(centre, seed=2)
    assert _brackets(g, pts) == 0
    out = recover_grasps(g, pts, max_shift=0.05)
    assert out, "expected at least one bracketing recovery"
    best = out[0]
    assert best.info["recovered"] is True
    assert best.info["recover_bracket"] > 0
    assert _brackets(best, pts) > 0
    assert np.linalg.norm(best.T[:2, 3] - centre[:2]) < np.linalg.norm(t[:2] - centre[:2])


def test_insertion_is_capped_and_never_dives():
    """Insertion along the approach axis never exceeds max_depth."""
    R = _topdown_R()
    t = np.array([0.5, 0.0, 0.25])
    g = _grasp(R, t)
    pts = _blob([0.5, 0.0, 0.10], seed=3)
    out = recover_grasps(g, pts, max_depth=0.015, max_shift=0.05)
    for gg in out:
        depth = float(gg.info["recover_shift"][2])
        assert 0.0 <= depth <= 0.015 + 1e-9


def test_results_ranked_and_capped_to_k():
    R = _topdown_R()
    g = _grasp(R, [0.5, 0.0, 0.15])
    pts = _blob([0.5, 0.0, 0.15], seed=4, n=600, spread=0.02)   # broad, so several offsets bracket it
    out = recover_grasps(g, pts, k=3, max_shift=0.05)
    assert 0 < len(out) <= 3
    brs = [gg.info["recover_bracket"] for gg in out]
    assert brs == sorted(brs, reverse=True)


def test_does_not_mutate_input():
    R = _topdown_R()
    t = np.array([0.5, 0.0, 0.15])
    g = _grasp(R, t)
    T0 = g.T.copy()
    recover_grasps(g, _blob([0.5, 0.0, 0.15], seed=5))
    assert np.allclose(g.T, T0)
    assert "recovered" not in g.info


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
