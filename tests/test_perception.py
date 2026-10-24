"""Unit tests for safegrasp.perception."""
import time

import numpy as np
import pytest
from scipy.spatial import cKDTree

from safegrasp.perception.types import View, SEG_ROBOT
from safegrasp.perception.pointcloud import (
    add_depth_noise, backproject, backproject_views, estimate_normals, depth_edges)
from safegrasp.perception.voxel import (
    FREE, OCC, UNK, VoxelGrid, build_occupancy, DistanceField, unknown_mask_for_target,
    obstacle_mask, label_counts)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """Return T_wc in the ROS optical frame (z forward, x right, y down)."""
    eye, target, up = (np.asarray(a, float) for a in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def intrinsics(W=640, H=480, f=600.0):
    return np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1.0]])


class Scene:
    """Table plane z = 0, boxes (centre, half extents, R, id) and spheres (centre, r, id)."""

    def __init__(self, boxes=(), spheres=(), table=True):
        self.boxes = [(np.asarray(c, float), np.asarray(h, float),
                       np.eye(3) if R is None else np.asarray(R, float), i)
                      for (c, h, R, i) in boxes]
        self.spheres = [(np.asarray(c, float), float(r), i) for (c, r, i) in spheres]
        self.table = table

    def render(self, K, T_wc, H=480, W=640):
        """Return depth (H, W) float32 z_cam, inf on a miss, and seg (H, W) int32."""
        v, u = np.mgrid[0:H, 0:W]
        pix = np.stack([u.ravel(), v.ravel(), np.ones(H * W)], 1).astype(float)
        d_cam = pix @ np.linalg.inv(K).T                  # z component = 1 -> t = z_cam
        R, o = T_wc[:3, :3], T_wc[:3, 3]
        d = d_cam @ R.T
        best = np.full(H * W, np.inf)
        seg = np.full(H * W, -1, np.int32)
        if self.table:
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(d[:, 2] < 0, -o[2] / d[:, 2], np.inf)
            best = np.minimum(best, t)
        for c, h, Rb, i in self.boxes:                    # slab test in the box frame
            ol = Rb.T @ (o - c)
            dl = d @ Rb
            dl = np.where(np.abs(dl) < 1e-12, 1e-12, dl)
            t1, t2 = (-h - ol) / dl, (h - ol) / dl
            tn = np.max(np.minimum(t1, t2), 1)
            tf = np.min(np.maximum(t1, t2), 1)
            hit = (tf >= tn) & (tn > 0) & (tn < best)
            best[hit], seg[hit] = tn[hit], i
        for c, r, i in self.spheres:
            oc = o - c
            a = np.einsum("ni,ni->n", d, d)
            b = d @ oc
            cc = oc @ oc - r * r
            disc = b * b - a * cc
            with np.errstate(invalid="ignore"):
                t = (-b - np.sqrt(disc)) / a
            hit = (disc >= 0) & (t > 0) & (t < best)
            best[hit], seg[hit] = t[hit], i
        return best.reshape(H, W).astype(np.float32), seg.reshape(H, W)

    def view(self, eye, target=(0.55, 0.0, 0.0), H=480, W=640, name=""):
        K = intrinsics(W, H)
        T = look_at(eye, target)
        depth, seg = self.render(K, T, H, W)
        return View(depth=depth, K=K, T_wc=T, seg=seg, name=name)


# id 0, tall; faces sit mid-voxel on the 1 cm grid
BOX_A = ((0.555, 0.005, 0.0625), (0.0325, 0.0325, 0.0625), None, 0)
SPH = ((0.505, 0.155, 0.045), 0.045, 1)
# id 2, hidden behind box A from the main camera
BOX_H = ((0.465, 0.005, 0.025), (0.02, 0.02, 0.025), None, 2)

EYE_MAIN = (1.05, 0.0, 0.55)
EYE_LEFT = (0.55, 0.60, 0.55)
EYE_RIGHT = (0.55, -0.60, 0.55)


@pytest.fixture(scope="module")
def scene():
    return Scene(boxes=[BOX_A, BOX_H], spheres=[SPH])


@pytest.fixture(scope="module")
def views(scene):
    return [scene.view(EYE_MAIN, name="main"), scene.view(EYE_LEFT, name="left"),
            scene.view(EYE_RIGHT, name="right")]


@pytest.fixture(scope="module")
def grid():
    return VoxelGrid()


def _label_at(labels, inst, grid, p):
    i = grid.world_to_index(np.asarray(p, float)[None])[0]
    return labels[tuple(i)], inst[tuple(i)]


def test_view_and_grid_basics(views):
    v = views[0]
    assert v.depth.dtype == np.float32 and v.seg.dtype == np.int32
    assert np.allclose(v.T_cw @ v.T_wc, np.eye(4))
    assert np.allclose(v.cam_center, EYE_MAIN)
    R = v.T_wc[:3, :3]
    assert np.isclose(np.linalg.det(R), 1.0)
    g = VoxelGrid()
    assert g.shape == (70, 80, 40) and np.allclose(g.origin, [0.2, -0.4, 0.0])
    assert np.allclose(g.hi, [0.9, 0.4, 0.4])
    idx = np.array([[0, 0, 0], [69, 79, 39], [10, 20, 30]])
    assert np.array_equal(g.world_to_index(g.index_to_world(idx)), idx)
    c = g.centers()
    assert c.shape == (70, 80, 40, 3) and np.allclose(c[3, 4, 5], g.index_to_world([3, 4, 5]))
    g2 = VoxelGrid.from_bounds([0.2, -0.4, 0.0], [0.9, 0.4, 0.4], 0.01)
    assert g2.shape == g.shape
    assert not g.in_bounds(np.array([[70, 0, 0]]))[0]


def test_backproject_roundtrip(scene, views):
    v = views[0]
    pts, ids, pix = backproject(v)
    assert pts.shape[1] == 3 and len(ids) == len(pts) == len(pix)
    tab = ids == -1
    assert np.abs(pts[tab, 2]).max() < 1e-4
    c, r, _ = scene.spheres[0]
    s = ids == 1
    assert s.sum() > 500
    assert np.abs(np.linalg.norm(pts[s] - c, axis=1) - r).max() < 1e-4
    assert np.allclose(v.depth[pix[:, 1], pix[:, 0]], (pts - v.T_wc[:3, 3]) @ v.T_wc[:3, 2],
                       atol=1e-4)
    # stride 4 keeps about 1/16 of the pixels
    p2, _, pix2 = backproject(v, stride=4)
    assert np.all(pix2 % 4 == 0) and 0.05 < len(p2) / len(pts) < 0.08


def test_backproject_skips_robot(views):
    v = views[0]
    seg = v.seg.copy()
    seg[100:200, 100:300] = SEG_ROBOT
    vr = View(v.depth, v.K, v.T_wc, seg)
    p_all, _, _ = backproject(vr, include_robot=True)
    p, ids, _ = backproject(vr)
    assert len(p_all) - len(p) == 100 * 200 and not np.any(ids == SEG_ROBOT)


def test_depth_noise_statistics():
    rng = np.random.default_rng(0)
    H, W, z0 = 240, 320, 0.8
    depth = np.full((H, W), z0, np.float32)
    n = add_depth_noise(depth, rng, sigma0=0.001, sigma2=0.002, dropout=0.0, edge_dropout=False)
    sig = 0.001 + 0.002 * z0 ** 2
    assert abs(np.std(n - z0) / sig - 1) < 0.05 and abs(np.mean(n - z0)) < 3e-4
    a = add_depth_noise(depth, np.random.default_rng(7), dropout=0.1)
    b = add_depth_noise(depth, np.random.default_rng(7), dropout=0.1)
    assert np.array_equal(a, b)
    assert abs(np.mean(a == 0) - 0.1) < 0.01
    # inf becomes 0; depth-step pixels are dropped
    step = np.full((H, W), 0.8, np.float32)
    step[:, W // 2:] = 0.9
    step[:10] = np.inf
    e = add_depth_noise(step, np.random.default_rng(1), edge_dropout=True)
    assert np.all(e[:10] == 0)
    assert np.all(e[10:, W // 2 - 1:W // 2 + 1] == 0)
    assert np.all(e[10:, :W // 2 - 1] > 0)
    assert depth_edges(step).sum() == 2 * (H - 10)


def test_estimate_normals(scene, views):
    v = views[0]
    pts, ids, _ = backproject(v, stride=2)
    tab = pts[(ids == -1)][:3000]
    nt = estimate_normals(tab, k=16, viewpoint=v.cam_center)
    assert np.mean(nt[:, 2] > 0.99) > 0.98
    c, r, _ = scene.spheres[0]
    sp = pts[ids == 1]
    ns = estimate_normals(sp, k=16, viewpoint=v.cam_center)
    ang = np.degrees(np.arccos(np.clip(np.sum(ns * (sp - c) / r, 1), -1, 1)))
    assert np.median(ang) < 3 and np.percentile(ang, 95) < 10
    P, I, VP = backproject_views(views, stride=3)
    sp = P[I == 1]
    ns = estimate_normals(sp, k=16, viewpoint=VP[I == 1])
    assert np.mean(np.sum(ns * (sp - c), 1) > 0) > 0.99
    # no viewpoint: normals default to +z
    sub = tab[np.random.default_rng(0).choice(len(tab), 200, replace=False)]
    assert np.all(estimate_normals(sub, k=8)[:, 2] > 0.9)
    assert estimate_normals(np.zeros((0, 3))).shape == (0, 3)


def test_occupancy_single_view(views, grid):
    labels, inst = build_occupancy(views[:1], grid)
    assert labels.shape == grid.shape and labels.dtype == np.int8 and inst.dtype == np.int32
    # in front of and above box A
    assert _label_at(labels, inst, grid, (0.625, 0.005, 0.075))[0] == FREE
    assert _label_at(labels, inst, grid, (0.555, 0.005, 0.185))[0] == FREE
    # front face and top of box A
    for p in [(0.585, 0.005, 0.065), (0.555, 0.005, 0.12)]:
        lab, i = _label_at(labels, inst, grid, p)
        assert lab == OCC and i == 0, (p, lab, i)
    c, r, _ = SPH
    p = np.array(c) + r * 0.999 * np.array([0.8, 0.0, 0.6])
    lab, i = _label_at(labels, inst, grid, p)
    assert lab == OCC and i == 1
    # behind box A, and inside it
    assert _label_at(labels, inst, grid, (0.495, 0.005, 0.045))[0] == UNK
    assert _label_at(labels, inst, grid, (0.555, 0.005, 0.045))[0] == UNK
    assert not np.any((labels == OCC) & (inst == 2))
    # the bottom layer over open table is carved
    assert _label_at(labels, inst, grid, (0.705, -0.205, 0.005))[0] == FREE
    assert not np.any((labels == OCC) & (inst == -1))
    # outside the camera view
    assert _label_at(labels, inst, grid, (0.205, -0.395, 0.395))[0] == UNK


def test_occupancy_multiview_reduces_unknown(views, grid):
    l1, _ = build_occupancy(views[:1], grid)
    l3, i3 = build_occupancy(views, grid)
    u1, u3 = label_counts(l1)["unk"], label_counts(l3)["unk"]
    assert u3 < 0.8 * u1, (u1, u3)
    assert np.any((l3 == OCC) & (i3 == 2))
    # centre of box A
    assert _label_at(l3, i3, grid, (0.555, 0.005, 0.06))[0] != FREE


def test_occupancy_robot_pixels(scene, grid):
    v = scene.view(EYE_MAIN)
    seg = v.seg.copy()
    # robot pixels cover box A in the image
    rows, cols = np.nonzero(v.seg == 0)
    r0, r1, c0, c1 = rows.min() - 5, rows.max() + 5, cols.min() - 5, cols.max() + 5
    seg[r0:r1, c0:c1] = SEG_ROBOT
    vr = View(v.depth, v.K, v.T_wc, seg)
    labels, inst = build_occupancy([vr], grid)
    assert not np.any((labels == OCC) & (inst == 0))
    assert _label_at(labels, inst, grid, (0.625, 0.005, 0.075))[0] == UNK
    labels2, _ = build_occupancy([vr], grid, carve_robot=True)
    assert _label_at(labels2, inst, grid, (0.625, 0.005, 0.075))[0] == FREE


def test_occupancy_with_noise(scene, grid):
    rng = np.random.default_rng(3)
    vs = []
    for eye in (EYE_MAIN, EYE_LEFT, EYE_RIGHT):
        v = scene.view(eye)
        vs.append(View(add_depth_noise(v.depth, rng, dropout=0.02), v.K, v.T_wc, v.seg))
    labels, inst = build_occupancy(vs, grid)
    assert np.sum((labels == OCC) & (inst == -1)) == 0
    for oid in (0, 1, 2):
        assert np.sum((labels == OCC) & (inst == oid)) > 10
    # min_z clears the table layer
    lab2, _ = build_occupancy(vs, grid, min_z=0.012)
    assert np.all(lab2[:, :, 0] == FREE)


def test_background_obstacle_marked(grid):
    # unsegmented box (seg -1)
    sc = Scene(boxes=[((0.605, 0.105, 0.05), (0.03, 0.03, 0.05), None, -1)])
    labels, inst = build_occupancy([sc.view(EYE_MAIN)], grid)
    lab, i = _label_at(labels, inst, grid, (0.625, 0.105, 0.095))
    assert lab == OCC and i == -1
    assert obstacle_mask(labels, inst, target_id=0)[tuple(grid.world_to_index(
        np.array([[0.625, 0.105, 0.095]]))[0])]


def test_unknown_mask_for_target(views, grid):
    labels, inst = build_occupancy(views[:1], grid)
    unk = labels == UNK
    m = unknown_mask_for_target(labels, inst, 0, grid, carve_radius=0.03)
    assert m.sum() < unk.sum() and not np.any(m & ~unk)
    # UNK at the target is carved, far UNK is kept
    assert not m[tuple(grid.world_to_index(np.array([[0.555, 0.005, 0.105]]))[0])]
    tgt_pts = grid.centers()[(labels == OCC) & (inst == 0)]
    dmin, _ = cKDTree(tgt_pts).query(grid.centers()[m])
    assert dmin.min() > 0.03 - 1e-9
    assert np.any(unk & ~m)
    T = np.eye(4)
    T[:3, 3] = (0.40, 0.0, 0.05)
    m2 = unknown_mask_for_target(labels, inst, 0, grid, carve_radius=0.03,
                                 extra_boxes=[(T, np.array([0.05, 0.05, 0.05]))])
    c = grid.centers()
    inbox = np.all(np.abs(c - T[:3, 3]) <= 0.05, axis=-1)
    assert not np.any(m2 & inbox) and np.any(m & inbox)
    assert np.array_equal(m2 | (m & inbox), m)
    assert np.array_equal(unknown_mask_for_target(labels, inst, 99, grid), unk)


@pytest.fixture(scope="module")
def sphere_field(grid):
    c, R = np.array([0.55, 0.0, 0.2]), 0.08
    mask = np.linalg.norm(grid.centers() - c, axis=-1) <= R
    return DistanceField.from_mask(mask, grid, max_dist=0.3), c, R


def test_distance_field_sphere(sphere_field, grid):
    df, c, R = sphere_field
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(4000, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rad = rng.uniform(0.0, 0.19, 4000)
    p = c + dirs * rad[:, None]
    d, g = df.query(p)
    sd = rad - R
    # under 1 voxel off, except near the centre where interpolation underestimates
    ok = rad > 2 * grid.res
    assert np.abs(d - sd)[ok].max() < grid.res, np.abs(d - sd)[ok].max()
    assert np.abs(d - sd).max() < 1.5 * grid.res
    sel = (np.abs(sd) > 1.5 * grid.res) & (rad > 0.04)
    cosang = np.sum(g[sel] * dirs[sel], axis=1)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    assert ang.max() < 15, ang.max()
    assert np.allclose(np.linalg.norm(g[sel], axis=1), 1.0)
    assert np.all(d[rad < R - 0.015] < 0)
    d1, g1 = df.query(c + np.array([0.15, 0, 0]))
    assert np.ndim(d1) == 0 and g1.shape == (3,)
    d2, g2 = df.query(p[:12].reshape(3, 4, 3))
    assert d2.shape == (3, 4) and g2.shape == (3, 4, 3)


def test_distance_field_outside_grid_lower_bound(sphere_field, grid):
    df, c, R = sphere_field
    rng = np.random.default_rng(1)
    p = rng.uniform(grid.lo - 0.4, grid.hi + 0.4, size=(5000, 3))
    p = p[~grid.contains(p)]
    d, g = df.query(p)
    true = np.linalg.norm(p - c, axis=1) - R
    assert np.all(d <= np.minimum(true, df.max_dist) + 0.4 * grid.res)
    box = np.linalg.norm(p - np.clip(p, grid.lo, grid.hi), axis=1)
    assert np.all(d >= np.minimum(box, df.max_dist) - 1e-12)
    assert np.allclose(np.linalg.norm(g, axis=1)[d < df.max_dist], 1.0)
    # slab touches the grid edge
    mask = np.zeros(grid.shape, bool)
    mask[30:40, 0:3, 0:10] = True                     # slab at y = -0.40 .. -0.37
    df2 = DistanceField.from_mask(mask, grid)
    q = np.array([[0.55, -0.45, 0.05], [0.55, -0.70, 0.05]])
    d2, g2 = df2.query(q)
    assert np.allclose(d2, [0.05, 0.30], atol=1e-9)
    assert np.allclose(g2, [[0, -1, 0], [0, -1, 0]], atol=1e-6)
    q = np.array([[0.55, -0.35, 0.45]])                 # above the grid, next to the slab
    d3, _ = df2.query(q)
    true3 = np.linalg.norm([0.02, 0.35])
    assert d3[0] <= true3 + 1e-9 and d3[0] >= 0.05 - 1e-9


def test_distance_field_empty_and_unsigned(grid):
    df = DistanceField.from_mask(np.zeros(grid.shape, bool), grid, max_dist=0.25)
    d, g = df.query(np.array([[0.5, 0.0, 0.1], [2.0, 2.0, 2.0]]))
    assert np.all(d == 0.25) and np.all(g == 0)
    mask = np.zeros(grid.shape, bool)
    mask[30:40, 30:40, 0:10] = True
    du = DistanceField.from_mask(mask, grid, signed=False)
    ds = DistanceField.from_mask(mask, grid, signed=True)
    inside = grid.index_to_world(np.array([[35, 35, 5]]))
    assert du.query(inside)[0][0] == 0.0 and ds.query(inside)[0][0] < -0.03
    far, _ = ds.query(np.array([[0.89, 0.39, 0.39]]))
    assert far[0] <= ds.max_dist


def test_distance_field_planar_exact(grid):
    """For a half-space mask the zero level sits exactly on the boundary face."""
    mask = np.zeros(grid.shape, bool)
    mask[:, :, :10] = True                             # z < 0.10
    df = DistanceField.from_mask(mask, grid)
    z = np.linspace(0.02, 0.3, 57)
    p = np.stack([np.full_like(z, 0.51), np.full_like(z, 0.03), z], 1)
    d, g = df.query(p)
    assert np.allclose(d, np.clip(z - 0.10, -0.3, 0.3), atol=1e-9)
    assert np.allclose(g[:, 2], 1.0)


def test_timing(views, grid, sphere_field):
    vs = [View(add_depth_noise(v.depth, np.random.default_rng(i)), v.K, v.T_wc, v.seg)
          for i, v in enumerate(views)]
    build_occupancy(vs, grid)                          # warm-up
    t = []
    for _ in range(3):
        t0 = time.perf_counter()
        build_occupancy(vs, grid)
        t.append(time.perf_counter() - t0)
    t_occ = min(t)
    df, c, R = sphere_field
    p = np.random.default_rng(0).uniform(grid.lo - 0.05, grid.hi + 0.05, size=(1000, 3))
    df.query(p)
    tq = []
    for _ in range(50):
        t0 = time.perf_counter()
        df.query(p)
        tq.append(time.perf_counter() - t0)
    t_q = min(tq)
    labels, inst = build_occupancy(vs, grid)
    t0 = time.perf_counter()
    DistanceField.from_mask(labels == UNK, grid)
    t_df = time.perf_counter() - t0
    print(f"\n[timing] build_occupancy 3x640x480: {t_occ * 1e3:.1f} ms | "
          f"DistanceField.from_mask: {t_df * 1e3:.1f} ms | query 1000 pts: {t_q * 1e6:.0f} us")
    assert t_occ < 0.3
    assert t_q < 1e-3


def test_occupancy_options(views, grid):
    labels, inst = build_occupancy(views[:1], grid)
    c = np.array([[0.495, 0.005, 0.045]])              # a UNK voxel (behind box A)
    assert _label_at(labels, inst, grid, c[0])[0] == UNK
    l2, _ = build_occupancy(views[:1], grid, robot_spheres=(c, np.array([0.02])), robot_pad=0.0)
    assert _label_at(l2, inst, grid, c[0])[0] == FREE
    assert np.array_equal(l2 == OCC, labels == OCC)
    # point_stride / min_points only thin the OCC set
    l3, i3 = build_occupancy(views[:1], grid, point_stride=3)
    l4, _ = build_occupancy(views[:1], grid, min_points=20)
    for lx in (l3, l4):
        assert np.all(labels[lx == OCC] == OCC) and 0 < np.sum(lx == OCC) <= np.sum(labels == OCC)
    assert np.sum(l3 == OCC) > 0.7 * np.sum(labels == OCC)
    lo, hi = np.array([0.35, -0.3, 0.0]), np.array([0.75, 0.3, 0.3])
    m = unknown_mask_for_target(labels, inst, 0, grid, roi=(lo, hi))
    cm = grid.centers()[m]
    assert len(cm) > 0 and np.all((cm >= lo) & (cm <= hi))


def test_dropout_does_not_create_unknown_speckles(scene, views, grid):
    """Dropped pixels leave no stray UNK voxels in seen free space."""
    v = views[0]
    rng = np.random.default_rng(5)
    vn = View(add_depth_noise(v.depth, rng, dropout=0.05), v.K, v.T_wc, v.seg)
    c = grid.centers()
    region = ((c[..., 0] > 0.62) & (c[..., 0] < 0.85) & (np.abs(c[..., 1]) < 0.15)
              & (c[..., 2] > 0.15) & (c[..., 2] < 0.3))
    l_clean, _ = build_occupancy([v], grid)
    assert np.all(l_clean[region] == FREE)
    l_nofill, _ = build_occupancy([vn], grid, fill_radius=0)
    l_fill, i_fill = build_occupancy([vn], grid)
    n0, n1 = np.sum(l_nofill[region] == UNK), np.sum(l_fill[region] == UNK)
    assert n0 > 20 and n1 == 0, (n0, n1)
    # the fill never frees voxels inside or behind box A
    for p in [(0.555, 0.005, 0.06), (0.495, 0.005, 0.045)]:
        assert _label_at(l_fill, i_fill, grid, p)[0] == UNK
