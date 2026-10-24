"""Unit tests for safegrasp.grasping."""
import json
import time

import numpy as np
import pytest

from safegrasp.grasping import antipodal as ant
from safegrasp.grasping import gripper as grp
from safegrasp.grasping import topdown as td
from safegrasp.grasping.types import (Grasp, MAX_WIDTH, TCP_OFFSET, TOPDOWN_QUAT, R_from_quat,
                                      canonicalize_about_approach, dedup_grasps, from_dict,
                                      hand_from_tcp, make_pose, pregrasp_pose, quat_from_R,
                                      rotation_angle_between, tcp_from_hand, to_dict)

CAM = np.array([1.15, 0.0, 0.65])      # front-top camera (robot base at origin)
GEOM = grp.get_geometry()


# Synthetic shapes: each returns points (N, 3) and outward normals (N, 3).
def _yaw(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def box_surface(center_xy, size, yaw=0.0, density=0.002, z0=0.0, bottom=False):
    """Box of size (sx, sy, sz) on z = z0, centred at center_xy."""
    sx, sy, sz = size
    P, N = [], []
    faces = [((0, 0, 1), (sx, sy), sz / 2), ((1, 0, 0), (sy, sz), sx / 2), ((-1, 0, 0), (sy, sz), sx / 2),
             ((0, 1, 0), (sx, sz), sy / 2), ((0, -1, 0), (sx, sz), sy / 2)]
    if bottom:
        faces.append(((0, 0, -1), (sx, sy), sz / 2))
    for n, (a, b), off in faces:
        n = np.array(n, float)
        na, nb = int(np.ceil(a / density)), int(np.ceil(b / density))
        u, v = np.meshgrid((np.arange(na) + 0.5) / na - 0.5, (np.arange(nb) + 0.5) / nb - 0.5, indexing="ij")
        u, v = u.ravel() * a, v.ravel() * b
        if n[2] != 0:
            p = np.stack([u, v, np.full_like(u, off * n[2])], 1)
        elif n[0] != 0:
            p = np.stack([np.full_like(u, off * n[0]), u, v], 1)
        else:
            p = np.stack([u, np.full_like(u, off * n[1]), v], 1)
        P.append(p)
        N.append(np.repeat(n[None], len(p), 0))
    P, N = np.concatenate(P), np.concatenate(N)
    R = _yaw(yaw)
    P = P @ R.T + np.array([center_xy[0], center_xy[1], z0 + sz / 2])
    return P, N @ R.T


def cylinder_surface(center_xy, radius, height, density=0.002):
    na = int(np.ceil(2 * np.pi * radius / density))
    nz = int(np.ceil(height / density))
    th, z = np.meshgrid(np.arange(na) * 2 * np.pi / na, (np.arange(nz) + 0.5) / nz * height, indexing="ij")
    th, z = th.ravel(), z.ravel()
    side = np.stack([radius * np.cos(th), radius * np.sin(th), z], 1)
    nside = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1)
    g = np.arange(-radius, radius, density) + density / 2
    X, Y = np.meshgrid(g, g, indexing="ij")
    m = X ** 2 + Y ** 2 <= radius ** 2
    top = np.stack([X[m], Y[m], np.full(m.sum(), height)], 1)
    P = np.concatenate([side, top]) + np.array([center_xy[0], center_xy[1], 0.0])
    N = np.concatenate([nside, np.repeat([[0, 0, 1.0]], len(top), 0)])
    return P, N


def sphere_surface(center, radius, density=0.002):
    n = int(4 * np.pi * radius ** 2 / density ** 2)
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    N = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    P = np.asarray(center) + radius * N
    keep = P[:, 2] > 0.002
    return P[keep], N[keep]


def mug_surface(center_xy, r_out=0.04, wall=0.005, height=0.09, handle_yaw=np.pi / 2, density=0.002):
    """Open mug: outer and inner walls, rim, inner floor and a handle."""
    Po, No = cylinder_surface((0, 0), r_out, height, density)
    side = No[:, 2] == 0
    Po, No = Po[side], No[side]
    r_in = r_out - wall
    Pi, Ni = cylinder_surface((0, 0), r_in, height - 0.008, density)
    side_i = Ni[:, 2] == 0
    floor = ~side_i
    Pi_side = Pi[side_i] + np.array([0, 0, 0.008])
    Ni_side = -Ni[side_i]
    Pf = Pi[floor] * np.array([1, 1, 0]) + np.array([0, 0, 0.008])
    Nf = np.repeat([[0, 0, 1.0]], len(Pf), 0)
    rr = np.arange(r_in, r_out, density / 2)
    th = np.arange(0, 2 * np.pi, density / r_out)
    RR, TH = np.meshgrid(rr, th, indexing="ij")
    Pr = np.stack([RR.ravel() * np.cos(TH.ravel()), RR.ravel() * np.sin(TH.ravel()), np.full(RR.size, height)], 1)
    Nr = np.repeat([[0, 0, 1.0]], len(Pr), 0)
    # handle: half a torus just outside the wall
    Rm, rm = 0.025, 0.006
    u, v = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, 60), np.linspace(0, 2 * np.pi, 24, endpoint=False), indexing="ij")
    u, v = u.ravel(), v.ravel()
    cxz = np.stack([Rm * np.cos(u), Rm * np.sin(u)], 1)          # centreline in (radial, z)
    nrm = np.stack([np.cos(u) * np.cos(v), np.sin(v), np.sin(u) * np.cos(v)], 1)  # (radial, tangential, z)
    loc = np.stack([r_out - 0.004 + cxz[:, 0] + rm * nrm[:, 0], rm * nrm[:, 1], height / 2 + cxz[:, 1] + rm * nrm[:, 2]], 1)
    loc_n = nrm
    Rh = _yaw(handle_yaw)
    Ph = loc @ Rh.T
    Nh = loc_n @ Rh.T
    P = np.concatenate([Po, Pi_side, Pf, Pr, Ph])
    N = np.concatenate([No, Ni_side, Nf, Nr, Nh])
    return P + np.array([center_xy[0], center_xy[1], 0.0]), N


def observe(clouds, cam=CAM, ang_res=0.004, depth_tol=0.004):
    """Visible part of each (P, N) cloud from cam: back-face culling and a z-buffer."""
    allP = np.concatenate([c[0] for c in clouds])
    allN = np.concatenate([c[1] for c in clouds])
    lab = np.concatenate([np.full(len(c[0]), i) for i, c in enumerate(clouds)])
    v = allP - cam
    rng_ = np.linalg.norm(v, axis=1)
    d = v / rng_[:, None]
    front = np.sum(allN * d, axis=1) < 0
    az = np.arctan2(d[:, 1], d[:, 0])
    el = np.arcsin(np.clip(d[:, 2], -1, 1))
    key = np.floor(az / ang_res).astype(np.int64) * 1000003 + np.floor(el / ang_res).astype(np.int64)
    _, inv = np.unique(key, return_inverse=True)
    zbuf = np.full(inv.max() + 1, np.inf)
    np.minimum.at(zbuf, inv[front], rng_[front])
    vis = front & (rng_ <= zbuf[inv] + depth_tol)
    return [(allP[vis & (lab == i)], allN[vis & (lab == i)]) for i in range(len(clouds))]


def split(clouds, target=0):
    vis = observe(clouds)
    tp, tn = vis[target]
    obs = [v[0] for i, v in enumerate(vis) if i != target]
    op = np.concatenate(obs) if obs else np.zeros((0, 3))
    return tp, tn, op


def check_valid(grasps, tp, op, margin=0.003):
    """Check invariants of proposer outputs."""
    assert len(grasps) > 0
    allp = np.concatenate([tp, op]) if len(op) else tp
    Ts = np.stack([g.T for g in grasps])
    ws = np.array([g.width for g in grasps])
    for g in grasps:
        R = g.T[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0.999
        assert 0.0 < g.width <= MAX_WIDTH + 1e-9
        assert g.T[2, 2] <= -0.2 + 1e-9                         # approach points downward
        assert 0.0 <= g.score <= 1.0
        json.dumps(to_dict(g))
    assert not np.any(grp.collides_batch(Ts, ws, allp, margin=margin))
    # also along the approach from the pre-grasp
    assert not np.any(grp.collides_batch(Ts, ws, allp, margin=margin, sweep=0.09))
    assert np.all(grp.lowest_z_batch(Ts, ws) >= 0.0)
    assert np.all(grp.lowest_z_batch(Ts, ws, fingers_only=True) >= 0.003 - 1e-9)
    scores = [g.score for g in grasps]
    assert scores == sorted(scores, reverse=True)


def test_types_roundtrip_and_pregrasp():
    R = R_from_quat(TOPDOWN_QUAT)
    assert np.allclose(R, np.diag([1, -1, -1]))
    T = make_pose(R, [0.5, 0.1, 0.05])
    g = Grasp(T, 0.05, 0.7, "x", {"a": np.float32(1.5)})
    Tp = pregrasp_pose(g, 0.12)
    assert np.allclose(Tp[:3, 3], [0.5, 0.1, 0.17])
    assert np.allclose(pregrasp_pose(np.stack([T, T]), 0.1)[:, 2, 3], 0.15)
    d = to_dict(g)
    s = json.dumps(d)
    g2 = from_dict(json.loads(s))
    assert np.allclose(g2.T, g.T) and g2.width == g.width and g2.info["a"] == 1.5
    assert np.allclose(d["quat_wxyz"], TOPDOWN_QUAT)
    Th = hand_from_tcp(T)
    assert np.isclose(Th[2, 3], 0.05 + TCP_OFFSET)
    assert np.allclose(tcp_from_hand(Th), T)
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] >= 0 else -q
        assert np.allclose(quat_from_R(R_from_quat(q)), q, atol=1e-9)


def test_canonicalize_and_dedup():
    R = R_from_quat(TOPDOWN_QUAT)
    T = make_pose(R, [0.5, 0.0, 0.05])
    Tf = T.copy()
    Tf[:3, :2] *= -1                              # same grasp rotated by pi about approach
    assert np.allclose(canonicalize_about_approach(Tf), T)
    assert np.isclose(rotation_angle_between(R, Tf[:3, :3], symmetric=True), 0.0)
    g1, g2 = Grasp(T, 0.04, 0.9, "x"), Grasp(Tf, 0.04, 0.8, "x")
    T3 = T.copy()
    T3[0, 3] += 0.03
    g3 = Grasp(T3, 0.04, 0.7, "x")
    out = dedup_grasps([g1, g2, g3])
    assert [g.score for g in out] == [0.9, 0.7]


def test_gripper_collision_basic():
    T = make_pose(R_from_quat(TOPDOWN_QUAT), [0.5, 0.0, 0.05])
    # hand y = -world y for the top-down quat
    between = np.array([[0.5, 0.0, 0.05]])
    finger = np.array([[0.5, 0.04 + 0.01, 0.05]])            # inside a finger when w = 0.08
    palm = np.array([[0.5, 0.0, 0.05 + 0.06]])
    above = np.array([[0.5, 0.0, 0.05 + 0.40]])
    assert not grp.collides(T, 0.08, between)
    assert grp.collides(T, 0.08, finger)
    assert not grp.collides(T, 0.02, finger)
    assert grp.collides(T, 0.08, palm)
    assert not grp.collides(T, 0.08, above)
    assert grp.collides(T, 0.08, above, sweep=0.40)          # hit during the approach
    assert grp.closing_region_mask(T, 0.08, between)[0]
    assert not grp.closing_region_mask(T, 0.08, palm)[0]
    assert grp.closing_region_points is grp.closing_region_mask
    rng = np.random.default_rng(1)
    pts = rng.uniform([0.4, -0.1, 0.0], [0.6, 0.1, 0.2], size=(3000, 3))
    Ts = []
    for i in range(20):
        Ri = R_from_quat(rng.normal(size=4))
        Ts.append(make_pose(Ri, rng.uniform([0.45, -0.05, 0.05], [0.55, 0.05, 0.15])))
    Ts = np.stack(Ts)
    ws = rng.uniform(0.0, 0.08, size=20)
    b = grp.collides_batch(Ts, ws, pts, margin=0.002)
    s = np.array([grp.collides(T_, w_, pts, margin=0.002) for T_, w_ in zip(Ts, ws)])
    assert np.array_equal(b, s)
    cm = grp.closing_region_mask_batch(Ts, ws, pts)
    assert np.array_equal(cm[3], grp.closing_region_mask(Ts[3], ws[3], pts))


def test_gripper_spheres_cover_boxes():
    for w in (0.0, 0.04, 0.08):
        c, r = grp.gripper_spheres(None, w)
        lo, hi = GEOM.boxes(w, include_wrist=False)
        rng = np.random.default_rng(0)
        for l, h in zip(lo, hi):
            P = rng.uniform(l, h, size=(5000, 3))
            gap = (np.linalg.norm(P[:, None] - c[None], axis=-1) - r[None]).min(1)
            assert gap.max() <= 1e-9
    T = make_pose(R_from_quat(TOPDOWN_QUAT), [0.5, 0.0, 0.05])
    cw, rw = grp.gripper_spheres(T, 0.08)
    c0, _ = grp.gripper_spheres(None, 0.08)
    assert np.allclose(cw, c0 @ T[:3, :3].T + T[:3, 3])


def test_voxel_downsample_covering_radius():
    rng = np.random.default_rng(3)
    P = np.concatenate([box_surface((0.5, 0.0), (0.05, 0.04, 0.06))[0], rng.uniform(0.3, 0.7, size=(2000, 3))])
    idx, rad, mem, start, cnt = grp.voxel_downsample(P, 0.005, return_groups=True)
    assert np.all(np.diff(idx) > 0) and cnt.sum() == len(P) and np.array_equal(np.sort(mem), np.arange(len(P)))
    for v in rng.choice(len(idx), 200, replace=False):
        members = P[mem[start[v]:start[v] + cnt[v]]]
        assert idx[v] in mem[start[v]:start[v] + cnt[v]]
        assert np.isclose(np.linalg.norm(members - P[idx[v]], axis=1).max(), rad[v])
        assert np.all(np.floor(members / 0.005) == np.floor(P[idx[v]] / 0.005))


def test_cloud_collider_matches_bruteforce():
    clouds = [box_surface((0.55, 0.0), (0.05, 0.05, 0.08)), box_surface((0.55, 0.075), (0.06, 0.05, 0.10)),
              mug_surface((0.45, -0.08))]
    P = np.concatenate([c[0] for c in clouds])
    rng = np.random.default_rng(5)
    Ts = np.stack([make_pose(R_from_quat(rng.normal(size=4)), rng.uniform([0.45, -0.1, 0.02], [0.6, 0.1, 0.15]))
                   for _ in range(150)])
    ws = rng.uniform(0.0, 0.08, size=150)
    for voxel in (0.004, 0.01):
        col = grp.CloudCollider(P, voxel)
        for margin, sweep in ((0.003, 0.0), (0.004, 0.1), (0.0, 0.05)):
            ref = grp.collides_batch(Ts, ws, P, margin=margin, sweep=sweep)
            assert 0 < ref.sum() < len(ref)
            assert np.array_equal(col.collides(Ts, ws, margin, sweep=sweep), ref)


def test_geometry_symmetrized_and_reload(tmp_path):
    g = grp.GripperGeometry.from_dict({"palm_lo": [-0.03, -0.10, -0.13], "palm_hi": [0.032, 0.105, -0.04],
                                       "finger_x": [-0.01, 0.012]})
    s = g.symmetrized()
    assert np.allclose(s.palm_lo[:2], [-0.032, -0.105]) and np.allclose(s.palm_hi[:2], [0.032, 0.105])
    assert np.allclose(s.finger_x, [-0.012, 0.012]) and np.allclose(g.palm_lo[:2], [-0.03, -0.10])
    # the Rz(pi) image of every box of g lies inside a box of s
    lo, hi = g.boxes(0.05)
    slo, shi = s.boxes(0.05)
    flip = np.array([-1.0, -1.0, 1.0])
    for l, h in zip(lo, hi):
        c = np.array([[x, y, z] for x in (l[0], h[0]) for y in (l[1], h[1]) for z in (l[2], h[2])]) * flip
        assert any(np.all((c >= a - 1e-12) & (c <= b + 1e-12)) for a, b in zip(slo, shi))
    # get_geometry picks up a file that appears or changes later
    path = str(tmp_path / "geom.json")
    assert grp.get_geometry(path).source == "defaults"
    with open(path, "w") as f:
        json.dump({"finger_thickness": 0.02}, f)
    assert np.isclose(grp.get_geometry(path).finger_thickness, 0.02)
    import os
    with open(path, "w") as f:
        json.dump({"finger_thickness": 0.021}, f)
    os.utime(path, ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))   # force a new mtime
    assert np.isclose(grp.get_geometry(path).finger_thickness, 0.021)


def test_geometry_from_dict():
    g = grp.GripperGeometry.from_dict({"palm": {"lo": [-0.03, -0.1, -0.12], "hi": [0.03, 0.1, -0.04]},
                                       "finger": {"x": [-0.01, 0.01], "y": [0.0, 0.02], "z": [-0.05, 0.01]},
                                       "pad": {"z": [-0.02, 0.004]}})
    assert np.isclose(g.palm_hi[2], -0.04) and np.isclose(g.finger_thickness, 0.02)
    assert np.isclose(g.pad_z[1], 0.004)
    lo, hi = g.finger_boxes(0.06)
    assert np.isclose(lo[0, 1], 0.03) and np.isclose(hi[1, 1], -0.03)


@pytest.mark.parametrize("yaw", [0.0, 0.5, 1.3, -0.8])
def test_topdown_box_short_side(yaw):
    P, N = box_surface((0.55, 0.05), (0.12, 0.05, 0.06), yaw=yaw)
    tp, tn, _ = split([(P, N)])
    gs = td.propose_topdown(tp, np.random.default_rng(0), None)
    g = gs[0]
    short_dir = _yaw(yaw)[:, 1]
    assert abs(g.closing @ short_dir) > 0.97
    assert np.allclose(g.approach, [0, 0, -1])
    assert 0.06 <= g.width <= 0.08
    assert np.isclose(g.position[2], 0.06 - 0.03, atol=0.004)
    assert np.linalg.norm(g.position[:2] - [0.55, 0.05]) < 0.012
    assert g.T[0, 0] * 0.55 + g.T[1, 0] * 0.05 >= 0              # canonical: hand x away from base
    assert len(gs) >= 3 and all(gs[i].score >= gs[i + 1].score for i in range(len(gs) - 1))


def test_topdown_flat_and_too_wide():
    P, N = box_surface((0.5, 0.0), (0.10, 0.03, 0.015), yaw=0.2)
    g = td.propose_topdown(split([(P, N)])[0], None, {"yaw_variants_deg": ()})[0]
    assert g.position[2] >= 0.012 - 1e-9 and g.position[2] <= 0.015
    P, N = box_surface((0.5, 0.0), (0.12, 0.10, 0.05))
    g = td.propose_topdown(split([(P, N)])[0], None, None)[0]
    assert g.score < 0.1 and g.width == pytest.approx(0.08)


def _scene_box_with_neighbours():
    tgt = box_surface((0.55, 0.0), (0.05, 0.05, 0.08), yaw=0.0)
    n1 = box_surface((0.55, 0.075), (0.06, 0.05, 0.10))
    n2 = box_surface((0.55, -0.075), (0.06, 0.05, 0.10))
    return [tgt, n1, n2]


def test_antipodal_box_valid():
    tp, tn, op = split([box_surface((0.55, 0.02), (0.05, 0.035, 0.07), yaw=0.4)])
    gs = ant.propose_antipodal(tp, tn, op, np.random.default_rng(0), None)
    check_valid(gs, tp, op)
    assert len(gs) <= 64
    info_keys = {"antipodal", "center", "n_region", "vertical", "depth", "w_req"}
    assert info_keys <= set(gs[0].info)
    # best grasp closes across the 3.5 cm side
    assert abs(gs[0].closing @ _yaw(0.4)[:, 1]) > 0.9


def test_antipodal_cylinder_axis():
    c = np.array([0.5, -0.05])
    tp, tn, op = split([cylinder_surface(c, 0.03, 0.10)])
    gs = ant.propose_antipodal(tp, tn, op, np.random.default_rng(1), None)
    check_valid(gs, tp, op)
    # the closing line of the top grasps passes near the cylinder axis
    for g in gs[:5]:
        p, y = g.position, g.closing
        n = np.cross(y, [0, 0, 1.0])
        nn = np.linalg.norm(n)
        d = abs((np.append(c, 0) - p) @ n) / nn if nn > 1e-6 else np.linalg.norm(p[:2] - c)
        assert d < 0.012, d
        assert g.width <= 0.08


def test_antipodal_sphere_and_mug():
    tp, tn, op = split([sphere_surface([0.5, 0.1, 0.03], 0.03)])
    gs = ant.propose_antipodal(tp, tn, op, np.random.default_rng(2), None)
    check_valid(gs, tp, op)
    mug = mug_surface((0.55, 0.0))
    tp, tn, op = split([mug])
    gs = ant.propose_antipodal(tp, tn, op, np.random.default_rng(3), None)
    check_valid(gs, tp, op)
    # the 8 cm mug is too wide: grasp the wall or handle
    r = np.linalg.norm(gs[0].position[:2] - [0.55, 0.0])
    assert r > 0.02


def test_obstacle_changes_closing_direction():
    # alone, both closing directions work
    tgt = box_surface((0.55, 0.0), (0.05, 0.05, 0.08))
    tp0, tn0, op0 = split([tgt])
    gs0 = ant.propose_antipodal(tp0, tn0, op0, np.random.default_rng(0), None)
    along_y0 = np.array([abs(g.closing[1]) for g in gs0])
    assert np.any(along_y0 > 0.95) and np.any(along_y0 < 0.3)
    # tall neighbours 2.5 cm away on both y sides block the fingers
    tp, tn, op = split(_scene_box_with_neighbours())
    gs = ant.propose_antipodal(tp, tn, op, np.random.default_rng(0), None)
    check_valid(gs, tp, op)
    along_y = np.array([abs(g.closing[1]) for g in gs])
    # diagonal grasps up to |closing_y| ~ 0.58 clear the neighbours but must rank low
    assert np.all(along_y < np.sqrt(0.5))
    assert np.all(along_y[:10] < 0.5)               # the grasps that matter close along x
    assert all(g.score < gs[0].score for g in gs if abs(g.closing[1]) >= 0.5)
    assert abs(gs[0].closing[0]) > 0.9


def test_antipodal_determinism_and_normals_optional():
    tp, tn, op = split(_scene_box_with_neighbours())
    a = ant.propose_antipodal(tp, tn, op, np.random.default_rng(7), None)
    b = ant.propose_antipodal(tp, tn, op, np.random.default_rng(7), None)
    assert len(a) == len(b)
    for ga, gb in zip(a, b):
        assert np.array_equal(ga.T, gb.T) and ga.score == gb.score and ga.width == gb.width
    c = ant.propose_antipodal(tp, None, op, np.random.default_rng(7), {"n_samples": 60})
    check_valid(c, tp, op)
    assert ant.propose_antipodal(tp[:5], tn[:5], op, 0, None) == []


def test_antipodal_one_sided_opens_fully():
    # closing along the view direction is one-sided
    tp, tn, op = split([cylinder_surface((0.55, 0.0), 0.028, 0.12)])
    gs = ant.propose_antipodal(tp, tn, op, np.random.default_rng(4), {"max_grasps": 200})
    one = [g for g in gs if g.info["one_sided"] > 0]
    for g in one:
        assert g.width == pytest.approx(0.08)
        assert g.info["antipodal"] < 0.75


def test_antipodal_speed():
    rng = np.random.default_rng(0)
    clouds = [cylinder_surface((0.55, 0.0), 0.03, 0.12, density=0.0015)]
    for k in range(9):
        ang = 2 * np.pi * k / 9
        c = (0.55 + 0.12 * np.cos(ang), 0.12 * np.sin(ang))
        clouds.append(box_surface(c, rng.uniform([0.04, 0.04, 0.05], [0.08, 0.08, 0.16]), yaw=ang, density=0.0022))
    tp, tn, op = split(clouds)
    it = rng.choice(len(tp), size=min(3000, len(tp)), replace=False)
    io = rng.choice(len(op), size=min(20000, len(op)), replace=False)
    tp, tn, op = tp[it], tn[it], op[io]
    ant.propose_antipodal(tp, tn, op, np.random.default_rng(0), None)     # warm-up
    t0 = time.perf_counter()
    gs = ant.propose_antipodal(tp, tn, op, np.random.default_rng(0), None)
    dt = time.perf_counter() - t0
    print(f"antipodal: {len(tp)} target + {len(op)} obstacle pts -> {len(gs)} grasps in {dt*1e3:.0f} ms")
    assert dt < 1.0            # aim 0.5 s; slack for busy machines
    check_valid(gs, tp, op)
