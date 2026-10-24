"""Edge-case and regression tests for safegrasp.perception."""
import warnings

import numpy as np
import pytest
from scipy.spatial import cKDTree

from safegrasp.perception.types import View, SEG_ROBOT
from safegrasp.perception.pointcloud import (
    add_depth_noise, backproject, estimate_normals)
from safegrasp.perception.voxel import (
    FREE, OCC, UNK, VoxelGrid, build_occupancy, DistanceField, unknown_mask_for_target,
    label_counts, _majority_vote)

from test_perception import Scene, EYE_MAIN, EYE_LEFT, EYE_RIGHT, BOX_A, SPH


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _random_clutter(seed, n=8):
    """Random yawed boxes and spheres on the table."""
    rng = np.random.default_rng(seed)
    boxes, sph = [], []
    for k in range(n):
        c = np.array([rng.uniform(0.38, 0.72), rng.uniform(-0.25, 0.25), 0.0])
        if rng.random() < 0.7:
            h = rng.uniform([0.015, 0.015, 0.02], [0.05, 0.03, 0.09])
            c[2] = h[2]
            boxes.append((c, h, _rotz(rng.uniform(0, np.pi)), k))
        else:
            r = rng.uniform(0.02, 0.05)
            c[2] = r
            sph.append((c, r, k))
    return Scene(boxes=boxes, spheres=sph)


def _scene_sdf(sc, P):
    """Return the exact signed distance to the objects, and the nearest id."""
    D, ID = [], []
    for c, h, R, i in sc.boxes:
        q = np.abs((P - c) @ R) - h
        D.append(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0))
        ID.append(i)
    for c, r, i in sc.spheres:
        D.append(np.linalg.norm(P - c, axis=1) - r)
        ID.append(i)
    D = np.stack(D, 1)
    return D.min(1), np.asarray(ID)[D.argmin(1)]


def _surface_samples(sc, rng, n=6000):
    P = []
    for c, h, R, _ in sc.boxes:
        u = rng.uniform(-1, 1, (n, 3))
        ax = rng.integers(0, 3, n)
        u[np.arange(n), ax] = np.sign(u[np.arange(n), ax])
        P.append(c + (u * h) @ R.T)
    for c, r, _ in sc.spheres:
        d = rng.normal(size=(n, 3))
        P.append(c + r * d / np.linalg.norm(d, axis=1, keepdims=True))
    P = np.concatenate(P)
    return P[P[:, 2] > 1e-4]


def _cube_union_distance(P, centres, res):
    """Return the exact distance from P (N, 3) to the union of cubes of side res."""
    q = np.abs(P[:, None, :] - centres[None]) - res / 2
    return np.linalg.norm(np.maximum(q, 0), axis=2).min(1)


@pytest.fixture(scope="module")
def grid():
    return VoxelGrid()


@pytest.fixture(scope="module")
def clutter_views():
    out = []
    for seed in (0, 3, 5):
        sc = _random_clutter(seed)
        vs = [sc.view(e) for e in (EYE_MAIN, EYE_LEFT, EYE_RIGHT)]
        rng = np.random.default_rng(100 + seed)
        noisy = [View(add_depth_noise(v.depth, rng, dropout=0.03), v.K, v.T_wc, v.seg) for v in vs]
        out.append((sc, vs, noisy))
    return out


def test_view_accepts_isaac_hw1_images(grid):
    """View squeezes Isaac Lab's (H, W, 1) images to (H, W)."""
    sc = Scene(boxes=[BOX_A])
    v = sc.view(EYE_MAIN)
    v1 = View(v.depth[..., None], v.K, v.T_wc, v.seg[..., None])
    assert v1.depth.shape == v.depth.shape and v1.seg.shape == v.seg.shape
    l0, _ = build_occupancy([v], grid)
    l1, _ = build_occupancy([v1], grid)
    assert np.array_equal(l0, l1)
    with pytest.raises(ValueError):
        View(np.zeros((4, 5, 3), np.float32), v.K, v.T_wc, None)


def test_voxelgrid_equality_and_centre_cache():
    a, b = VoxelGrid(), VoxelGrid()
    assert a == b and not (a != b)
    assert a != VoxelGrid(res=0.02) and a != VoxelGrid(origin=[0.2, -0.4, 0.01])
    g = VoxelGrid()
    c0 = g.centers()[0, 0, 0].copy()
    g.origin = g.origin + np.array([0.1, 0.0, 0.0])        # a new origin rebuilds the cache
    assert np.allclose(g.centers()[0, 0, 0], c0 + [0.1, 0, 0])
    g.origin[2] += 0.05                                     # so does an in-place edit
    assert np.allclose(g.centers()[0, 0, 0], c0 + [0.1, 0, 0.05])


def test_backproject_projects_back_to_its_pixels():
    sc = Scene(boxes=[BOX_A], spheres=[SPH])
    v = sc.view(EYE_LEFT)
    pts, _, pix = backproject(v, stride=3)
    pc = (pts - v.T_wc[:3, 3]) @ v.T_wc[:3, :3]             # camera frame
    assert np.all(pc[:, 2] > 0)                              # z forward
    uv = (pc @ v.K.T)[:, :2] / pc[:, 2:3]
    assert np.abs(uv - pix).max() < 1e-6
    assert np.allclose(pc[:, 2], v.depth[pix[:, 1], pix[:, 0]], atol=1e-5)


def test_depth_noise_edge_cases():
    rng = np.random.default_rng(0)
    d = np.array([[np.nan, np.inf, -np.inf, -0.3, 0.0, 0.7]], np.float32)
    out = add_depth_noise(d, rng)
    assert out.dtype == np.float32 and np.all(out[0, :5] == 0) and out[0, 5] > 0
    assert add_depth_noise(np.zeros((0, 0), np.float32), rng).shape == (0, 0)
    assert np.all(add_depth_noise(np.full((8, 8), 0.5, np.float32), rng, dropout=1.0) == 0)
    # std grows quadratically with range
    z = np.full((200, 200), 1.5, np.float32)
    s = np.std(add_depth_noise(z, rng, edge_dropout=False) - 1.5)
    assert abs(s / (0.001 + 0.002 * 1.5 ** 2) - 1) < 0.05
    z0 = z.copy()
    add_depth_noise(z, rng, dropout=0.5)
    assert np.array_equal(z, z0)


def test_estimate_normals_degenerate_neighbourhoods():
    vp = np.array([0.3, 0.2, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # coincident points: normal faces the viewpoint
        p = np.tile([[0.5, 0.0, 0.1]], (20, 1))
        n = estimate_normals(p, 8, viewpoint=vp)
        r = (vp - p[0]) / np.linalg.norm(vp - p[0])
        assert np.allclose(n, r)
        # collinear points
        x = np.linspace(0.4, 0.6, 30)
        line = np.stack([x, np.zeros_like(x), np.full_like(x, 0.1)], 1)
        n = estimate_normals(line, 8, viewpoint=vp)
        assert np.allclose(np.linalg.norm(n, axis=1), 1)
        assert np.abs(n[:, 0]).max() < 1e-6
        assert np.all(np.einsum("ni,ni->n", n, vp - line) > 0)
        # k < 3
        n = estimate_normals(line[:2], 16, viewpoint=vp)
        assert np.all(np.einsum("ni,ni->n", n, vp - line[:2]) > 0)
        # NaN points get a zero normal
        rng = np.random.default_rng(0)
        plane = np.c_[rng.uniform(0, 0.1, (200, 2)), np.zeros(200)]
        bad = plane.copy()
        bad[[3, 50]] = np.nan
        nb = estimate_normals(bad, 10, viewpoint=vp)
        ok = np.ones(200, bool)
        ok[[3, 50]] = False
        assert np.all(nb[~ok] == 0) and np.all(nb[ok, 2] > 0.999)
        vps = np.tile(vp, (200, 1))
        vps[:100] = [0.05, 0.05, -2.0]                        # below the plane -> normals down
        nb = estimate_normals(bad, 10, viewpoint=vps)
        assert np.all(nb[:100][ok[:100], 2] < -0.999) and np.all(nb[100:][ok[100:], 2] > 0.999)


def test_carving_never_frees_object_interior(clutter_views, grid):
    """No FREE voxel centre lies inside an object (2 mm slack for pixel rounding)."""
    C = grid.centers(flat=True)
    for sc, clean, noisy in clutter_views:
        sd, _ = _scene_sdf(sc, C)
        for vs in (clean[:1], clean, noisy[:1], noisy):
            L, _ = build_occupancy(vs, grid)
            assert sd[L.ravel() == FREE].min() > -0.002


def test_object_surfaces_covered_by_occ_or_unk(clutter_views, grid):
    """Surface points lie within ~half a voxel of a non-FREE cube."""
    rng = np.random.default_rng(0)
    for sc, clean, noisy in clutter_views:
        S = _surface_samples(sc, rng)
        S = S[grid.contains(S)]
        for vs in (clean[:1], noisy):
            L, _ = build_occupancy(vs, grid)
            nonfree = grid.centers()[L != FREE]
            # 32 neighbours cover every cube within res/2
            _, nn = cKDTree(nonfree).query(S, k=32)
            q = np.abs(S[:, None, :] - nonfree[nn]) - grid.res / 2
            gap = np.linalg.norm(np.maximum(q, 0), axis=2).min(1)
            assert gap.max() < 0.6 * grid.res, gap.max()


def test_occ_instance_ids_match_ground_truth(clutter_views, grid):
    C = grid.centers(flat=True)
    for sc, clean, noisy in clutter_views:
        sd, nid = _scene_sdf(sc, C)
        for vs in (clean, noisy):
            L, I = build_occupancy(vs, grid)
            occ = (L.ravel() == OCC)
            assert np.all(np.abs(sd[occ]) < 0.015)           # OCC voxels hug surfaces
            assert np.mean(I.ravel()[occ] == nid[occ]) > 0.97


def test_majority_vote_ties_and_min_points():
    lin = np.array([5, 5, 5, 5, 9, 9, 2])
    ids = np.array([-1, 3, -1, 3, 7, 7, -1])
    vox, lab = _majority_vote(lin, ids)
    assert dict(zip(vox.tolist(), lab.tolist())) == {2: -1, 5: 3, 9: 7}   # tie -> larger id
    vox, lab = _majority_vote(lin, ids, min_points=2)
    assert dict(zip(vox.tolist(), lab.tolist())) == {5: 3, 9: 7}


def test_robot_holes_not_filled_and_spheres_tuple(grid):
    """Invalid-depth robot pixels are not filled; robot_spheres takes Panda.spheres()'s 4-tuple."""
    sc = Scene(boxes=[BOX_A])
    v = sc.view(EYE_MAIN)
    depth, seg = v.depth.copy(), v.seg.copy()
    seg[200:300, 250:400] = SEG_ROBOT
    depth[200:300, 250:400] = 0.0
    vr = View(depth, v.K, v.T_wc, seg)
    L, _ = build_occupancy([vr], grid)
    pc_probe = np.array([0.0, 0.0, 0.6])                     # camera frame, in front of box A
    pw = v.T_wc[:3, :3] @ pc_probe + v.T_wc[:3, 3]
    uv = v.K @ pc_probe / pc_probe[2]
    assert 250 <= uv[0] < 400 and 200 <= uv[1] < 300 and grid.contains(pw[None])[0]
    L0, _ = build_occupancy([v], grid)
    assert L0[tuple(grid.world_to_index(pw[None])[0])] == FREE    # carved without the robot
    assert L[tuple(grid.world_to_index(pw[None])[0])] == UNK
    c = pw[None]
    L2, _ = build_occupancy([vr], grid, robot_spheres=(c, np.array([0.02]),
                                                        np.zeros((1, 3, 7)), np.zeros(1, int)))
    assert L2[tuple(grid.world_to_index(pw[None])[0])] == FREE


def test_inf_depth_and_points_outside_grid():
    """inf depth is carved only with inf_is_free; points outside the grid are ignored."""
    g = VoxelGrid(origin=[0.4, -0.1, 0.0], res=0.02, shape=(15, 10, 10))
    sc = Scene(boxes=[((0.55, 0.0, 0.1), (0.03, 0.03, 0.03), None, 4),
                      ((0.2, 0.3, 0.1), (0.05, 0.05, 0.05), None, 5)], table=False)
    v = sc.view(EYE_MAIN)
    assert np.isinf(v.depth).any()
    L, I = build_occupancy([v], g)
    assert set(np.unique(I[L == OCC]).tolist()) == {4}
    Lf, _ = build_occupancy([v], g, inf_is_free=True)
    assert label_counts(Lf)["free"] > label_counts(L)["free"]
    assert np.array_equal(Lf == OCC, L == OCC)


def test_unknown_mask_shadow_condition(grid):
    """With views, UNK near the target is removed only in the target's own shadow."""
    tgt = ((0.545, 0.005, 0.04), (0.02, 0.02, 0.04), None, 0)
    occluder = ((0.625, 0.065, 0.08), (0.02, 0.02, 0.08), None, 1)
    vs = [Scene(boxes=[tgt, occluder]).view(EYE_MAIN)]
    L, I = build_occupancy(vs, grid)
    m0 = unknown_mask_for_target(L, I, 0, grid)
    m1 = unknown_mask_for_target(L, I, 0, grid, views=vs)
    assert not np.any(m0 & ~m1)                              # views only ever keep more
    kept = m1 & ~m0
    assert kept.sum() >= 5
    C = grid.centers()
    assert np.all(C[kept][:, 1] > 0.04)                      # in the occluder's shadow
    # the target's own shadow is removed either way
    i = tuple(grid.world_to_index(np.array([[0.535, 0.005, 0.055]]))[0])
    assert L[i] == UNK and not m0[i] and not m1[i]
    T = np.eye(4)
    T[:3, 3] = (0.40, 0.0, 0.05)
    m2 = unknown_mask_for_target(L, I, 0, grid, extra_boxes=[(T, 0.05)])
    inbox = np.all(np.abs(C - T[:3, 3]) <= 0.05, axis=-1)
    assert not np.any(m2 & inbox) and np.any(m0 & inbox)


@pytest.fixture(scope="module")
def blob_fields():
    g = VoxelGrid(origin=[0.3, -0.2, 0.0], res=0.01, shape=(40, 40, 30))
    rng = np.random.default_rng(0)
    C = g.centers()
    out = []
    for _ in range(3):
        mask = np.zeros(g.shape, bool)
        for _ in range(3):
            c = rng.uniform(g.lo + 0.08, g.hi - 0.08)
            if rng.random() < 0.5:
                mask |= np.linalg.norm(C - c, axis=-1) < rng.uniform(0.02, 0.05)
            else:
                mask |= np.all(np.abs(C - c) < rng.uniform(0.01, 0.04, 3), axis=-1)
        out.append((DistanceField.from_mask(mask, g), C[mask]))
    return g, out


def test_distance_field_vs_exact_cube_distance(blob_fields):
    """The field is within +0.6 / -0.5 res of the exact distance to the mask cubes."""
    g, fields = blob_fields
    rng = np.random.default_rng(1)
    for df, cubes in fields:
        P = rng.uniform(g.lo + 0.01, g.hi - 0.01, (2000, 3))
        true = _cube_union_distance(P, cubes, g.res)
        d, _ = df.query(P)
        sel = (true > 0) & (true < 0.25)
        err = d[sel] - true[sel]
        assert err.max() < 0.6 * g.res, err.max()
        assert err.min() > -0.5 * g.res, err.min()
        assert np.all(d[true == 0] <= 0.5 * g.res)


def test_distance_field_gradient_matches_finite_differences(blob_fields):
    g, fields = blob_fields
    rng = np.random.default_rng(2)
    eps = 1e-4
    for df, cubes in fields:
        P = rng.uniform(g.lo + 0.01, g.hi - 0.01, (1500, 3))
        true = _cube_union_distance(P, cubes, g.res)
        _, gr = df.query(P)
        fd = np.stack([(df.query(P + eps * e)[0] - df.query(P - eps * e)[0]) / (2 * eps)
                       for e in np.eye(3)], 1)
        sel = (true > 0.015) & (true < 0.2) & (np.linalg.norm(fd, axis=1) > 0.5)
        cos = np.sum(gr[sel] * fd[sel], 1) / np.linalg.norm(fd[sel], axis=1)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        assert np.median(ang) < 3 and np.percentile(ang, 95) < 20, (np.median(ang), ang.max())
        assert np.median(np.linalg.norm(fd[sel], axis=1)) == pytest.approx(1.0, abs=0.05)


def test_distance_field_outside_grid_monotone_and_consistent(grid):
    """Leaving the grid, d stays a lower bound and never decreases."""
    mask = np.zeros(grid.shape, bool)
    mask[10:15, 35:45, 5:15] = True                          # x in [0.30, 0.35]
    df = DistanceField.from_mask(mask, grid)
    cubes = grid.centers()[mask]
    rng = np.random.default_rng(3)
    for _ in range(20):
        # walk outward from a random point on a grid face
        axis, side = rng.integers(0, 3), rng.integers(0, 2)
        p0 = rng.uniform(grid.lo, grid.hi)
        p0[axis] = grid.hi[axis] if side else grid.lo[axis]
        nrm = np.zeros(3)
        nrm[axis] = 1.0 if side else -1.0
        s = np.linspace(0.0, 0.3, 61)
        P = p0 + s[:, None] * nrm
        d, gr = df.query(P)
        assert np.all(np.diff(d) >= -1e-9)
        true = _cube_union_distance(P, cubes, grid.res)
        assert np.all(d <= np.minimum(true, df.max_dist) + 0.6 * grid.res)
        e = s
        assert np.all(d >= np.minimum(e, df.max_dist) - 1e-12)
        # never below d on the grid face
        assert np.all(d >= d[0] - 1e-12)
        # a step along the gradient never lowers d
        step = 1e-4
        d2, _ = df.query(P[1:] + step * gr[1:])
        assert np.all(d2 - d[1:] >= -1e-9)


def test_distance_field_single_voxel_axis():
    """A grid one voxel thick in z still has x/y gradients."""
    g = VoxelGrid(origin=[0.2, -0.4, 0.0], res=0.01, shape=(70, 80, 1))
    m = np.zeros(g.shape, bool)
    m[30:35, 30:35, 0] = True
    df = DistanceField.from_mask(m, g)
    d, gr = df.query(np.array([[0.60, 0.0, 0.005], [0.45, -0.1, 0.005]]))
    assert np.all(d > 0) and np.allclose(np.linalg.norm(gr, axis=1), 1.0)
    assert gr[0, 0] > 0.5 and gr[1, 0] < -0.5


def test_distance_field_nonfinite_query_is_quiet(grid):
    m = np.zeros(grid.shape, bool)
    m[30:40, 30:40, 0:10] = True
    df = DistanceField.from_mask(m, grid)
    P = np.array([[np.nan, 0.0, 0.1], [0.5, np.inf, 0.1], [0.55, 0.0, 0.2]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d, gr = df.query(P)
    assert np.isnan(d[0]) and np.isnan(d[1]) and np.isfinite(d[2])
    assert np.all(gr[:2] == 0) and np.isclose(np.linalg.norm(gr[2]), 1.0)
