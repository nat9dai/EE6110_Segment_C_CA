"""GPD-style antipodal grasp proposer on partial point clouds (distances in metres).

Samples hand frames on the target surface, finds the deepest collision-free insertion on each
approach line, scores antipodality (partial credit for unseen sides) and verifies the best exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .gripper import (CloudCollider, GripperGeometry, boxes_reach, closing_volume_voxel_counts, get_geometry,
                      voxel_downsample, voxel_labels)
from .types import DEFAULT_PREGRASP_DIST, Grasp, canonicalize_about_approach, dedup_grasps, make_cfg, make_pose


@dataclass
class AntipodalConfig:
    n_samples: int = 150
    tilts: tuple = (0.0, 0.35, 0.7, 1.0)     # slerp fraction  -normal -> -z_world
    n_rot: int = 8                           # closing-axis angles over [0, pi)
    random_rot_offset: bool = True
    depth_offsets: tuple = (0.0, 0.015, 0.03)
    max_depth: float = 0.06                  # TCP advance past the sampled point (along a)
    min_depth: float = -0.02
    approach_dist: float = DEFAULT_PREGRASP_DIST   # straight approach that must be collision free (m)
    margin: float = 0.004                    # clearance to every input point (m)
    search_inflation: float = 0.25           # lambda: search boxes inflated by margin + lambda*r_voxel
    verify: bool = True
    verify_voxel: float = 0.008              # broad-phase voxel of the verifier (speed only)
    table_z: float = 0.0
    table_clear: float = 0.003
    min_points: int = 15
    max_obstacle_in_region: int = 2          # obstacle voxels tolerated between the fingers (noise)
    width_pad: float = 0.015                 # w_req = extent + pad
    open_extra: float = 0.01                 # commanded width = w_req + extra (<= max_width)
    min_width: float = 0.01
    max_width: float = 0.08
    max_approach_z: float = -0.2             # approach must point at least a bit downward
    max_toward_base: float = 0.5             # reject a_xy . r_hat < -this
    max_shift: float = 0.03                  # max TCP re-centring along the closing axis
    base_xy: tuple = (0.0, 0.0)
    include_wrist: bool = True
    friction: float = 0.5
    band: float = 0.005
    facing_cos: float = 0.5                  # a band "faces" its pad if some n.(+-y) > this
    unobserved_credit: float = 0.35
    center_sigma: float = 0.02
    count_full: int = 40
    w_antipodal: float = 0.45
    w_center: float = 0.2
    w_count: float = 0.15
    w_vertical: float = 0.2
    voxel_target: float = 0.003
    voxel_obstacle: float = 0.005
    normal_k: int = 16
    max_grasps: int = 64
    dedup_pos: float = 0.01
    dedup_ang_deg: float = 15.0
    refine_top: int = 24                     # best frames refined locally (0 = off)
    refine_shifts: tuple = (-0.01, -0.005, 0.0, 0.005, 0.01)
    method: str = "antipodal"


def voxel_downsample_idx(pts: np.ndarray, voxel: float) -> np.ndarray:
    """Indices of one representative point per occupied voxel (sorted)."""
    return voxel_downsample(pts, voxel)[0]


def estimate_normals_pca(pts: np.ndarray, k: int = 16, center=None) -> np.ndarray:
    """PCA normals (smallest covariance eigenvector of k-NN), oriented away from ``center``."""
    k = int(min(k, len(pts)))
    _, nn = cKDTree(pts).query(pts, k=k)
    Q = pts[nn] - pts[nn].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Q, Q)
    _, V = np.linalg.eigh(C)
    n = V[:, :, 0]
    c = pts.mean(0) if center is None else np.asarray(center)
    n *= np.where(np.sum(n * (pts - c), axis=1) < 0, -1.0, 1.0)[:, None]
    return n


def _slerp(u0, u1, t):
    """Spherical interpolation between unit vectors (u0 not antipodal to u1)."""
    om = np.arccos(np.clip(np.dot(u0, u1), -1.0, 1.0))
    if om < 1e-6:
        return u1.copy()
    return (np.sin((1 - t) * om) * u0 + np.sin(t * om) * u1) / np.sin(om)


def _perp_basis(a):
    """(e1, e2) with (e1, e2, a) right-handed orthonormal."""
    h = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, h)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2


_INFO_KEYS = ("antipodal", "q_pos", "q_neg", "one_sided", "center", "center_dist", "n_region", "count",
              "vertical", "depth", "w_req", "y_shift", "tilt", "approach_z", "lowest_z")


def _diverse(cand, order, n, pos_tol, ang_tol):
    """First ``n`` indices of ``order`` that are pairwise apart (position or angle)."""
    from .types import rotation_angle_between
    out = []
    for i in order:
        if len(out) >= n:
            break
        if out:
            j = np.asarray(out)
            close = np.linalg.norm(cand["p"][j] - cand["p"][i], axis=1) < pos_tol
            if np.any(close):
                ang = rotation_angle_between(cand["R"][i][None], cand["R"][j[close]])
                if np.any(ang < ang_tol):
                    continue
        out.append(i)
    return out


class _LineEvaluator:
    """Evaluate all hand frames on one approach line (point s, direction a).

    C: (n,3) collision points, the first nt are target. infl: (n,) per-point box inflation (m).
    """

    def __init__(self, C, infl, nt, tn, center, cfg, geom):
        self.C, self.infl, self.nt, self.tn, self.center, self.cfg, self.geom = C, infl, nt, tn, center, cfg, geom
        m = float(infl.max()) if len(infl) else cfg.margin
        self.slo, self.shi = geom.static_boxes(cfg.include_wrist)
        flo, fhi = geom.finger_boxes(cfg.max_width)                     # fully open for the sweep
        self.blo = np.concatenate([self.slo, flo])                      # (B,3)
        self.bhi = np.concatenate([self.shi, fhi])
        # crop radius: inflated boxes with the whole hand shifted up to max_shift along y
        ext_x = np.maximum(np.abs(self.blo[:, 0] - m), np.abs(self.bhi[:, 0] + m))
        ext_y = np.maximum(np.abs(self.blo[:, 1] - m), np.abs(self.bhi[:, 1] + m)) + cfg.max_shift
        self.r_xy = float(np.hypot(ext_x, ext_y).max())
        self.fz_lo, self.fz_hi = geom.finger_z                          # finger length (closing sweep)
        self.z_lo = min(cfg.min_depth, 0.0) - cfg.approach_dist + self.blo[:, 2].min() - m
        self.z_hi = cfg.max_depth + self.bhi[:, 2].max() + m
        self.corners0 = geom.corners(cfg.max_width, cfg.include_wrist)
        fx = max(abs(geom.finger_x[0]), abs(geom.finger_x[1]))
        self.pad_xh = fx                                                 # pad half-width (x)
        self.cx_half = geom.closing_x_scale * fx                         # closing-region half-width
        self.pad_lo, self.pad_hi = geom.pad_z
        self.cphi = np.cos(np.arctan(cfg.friction))
        self.c2phi = np.cos(2.0 * np.arctan(cfg.friction))
        self.d_off = np.asarray(cfg.depth_offsets, dtype=np.float64)
        self.psis = np.arange(cfg.n_rot) * np.pi / cfg.n_rot

    def eval_line(self, s, a, psis, beta, stats):
        """Feasible candidates for TCP on the line s + d a with closing angles psis: dict of arrays, or None."""
        cfg, geom = self.cfg, self.geom
        e1, e2 = _perp_basis(a)
        V = self.C - s
        zl = V @ a
        r2 = np.einsum("ij,ij->i", V, V) - zl * zl
        nb = np.flatnonzero((r2 <= self.r_xy ** 2) & (zl >= self.z_lo) & (zl <= self.z_hi))
        V, zl, mi = V[nb], zl[nb], self.infl[nb]                  # mi (n,): per-point inflation
        is_t = nb < self.nt
        u1, u2 = V @ e1, V @ e2
        cs, sn = np.cos(psis)[:, None], np.sin(psis)[:, None]     # (J,1)
        xl = sn * u1 - cs * u2                                    # (J,n) hand-x coordinate
        yl = cs * u1 + sn * u2                                    # (J,n) closing coordinate
        J = len(psis)
        stats["n_frames"] += J

        # first contact depth per rotation; a point is in box b for d in [zl - hi_z - m_i, zl - lo_z + m_i]
        d_start = -cfg.approach_dist
        d_first = np.full(J, np.inf)
        blo, bhi = self.blo, self.bhi
        for b in range(len(blo)):
            live = zl - blo[b, 2] + mi >= d_start                 # not already passed at the start
            if not np.any(live):
                continue
            inb = ((xl >= blo[b, 0] - mi) & (xl <= bhi[b, 0] + mi)
                   & (yl >= blo[b, 1] - mi) & (yl <= bhi[b, 1] + mi) & live)
            lower = np.where(inb, (zl - bhi[b, 2] - mi)[None], np.inf)
            d_first = np.minimum(d_first, lower.min(1))
        # table: lowest corner z(d) = s_z + d a_z + min_c(R c)_z >= table + clear
        xz = sn[:, 0] * e1[2] - cs[:, 0] * e2[2]
        yz = cs[:, 0] * e1[2] + sn[:, 0] * e2[2]
        c0 = self.corners0
        off = (c0[:, 0][None] * xz[:, None] + c0[:, 1][None] * yz[:, None] + c0[:, 2][None] * a[2]).min(1)
        d_table = (s[2] + off - cfg.table_z - cfg.table_clear) / max(-a[2], 1e-6)
        d_hi = np.minimum(np.minimum(d_first - 1e-4, d_table), cfg.max_depth)
        okj = (d_first > d_start) & (d_hi >= cfg.min_depth)
        if not np.any(okj):
            return None
        # (J,D) depth candidates
        D =d_hi[:, None] - self.d_off[None]
        jj, dd = np.nonzero(okj[:, None] & (D >= cfg.min_depth))
        if jj.size == 0:
            return None
        dsel = D[jj, dd]                                          # (M,)
        stats["n_depth"] += jj.size

        # closing region at full opening
        zrel = zl[None] - dsel[:, None]                           # (M,n) TCP-frame z
        xm, ym = xl[jj], yl[jj]
        reg = ((np.abs(xm) <= self.cx_half) & (np.abs(ym) < 0.5 * cfg.max_width)
               & (zrel >= self.pad_lo) & (zrel <= self.pad_hi))
        reg_t = reg & is_t[None]
        n_reg = reg_t.sum(1)
        n_obs = (reg & ~is_t[None]).sum(1)
        good = (n_reg >= cfg.min_points) & (n_obs <= cfg.max_obstacle_in_region)
        if not np.any(good):
            return None
        stats["n_region"] += int(good.sum())
        jj, dsel, xm, ym, zrel, reg_t = (v[good] for v in (jj, dsel, xm, ym, zrel, reg_t))
        ymax = np.where(reg_t, ym, -np.inf).max(1)
        ymin = np.where(reg_t, ym, np.inf).min(1)
        w_req = ymax - ymin + cfg.width_pad
        good = w_req <= cfg.max_width
        if not np.any(good):
            return None
        stats["n_width"] += int(good.sum())
        jj, dsel, xm, ym, zrel, reg_t, ymax, ymin, w_req = (
            v[good] for v in (jj, dsel, xm, ym, zrel, reg_t, ymax, ymin, w_req))

        # antipodal quality on the pads; a side never seen gets partial credit
        tl = np.flatnonzero(is_t)
        nrm = self.tn[nb[tl]]
        ny = cs * (nrm @ e1)[None] + sn * (nrm @ e2)[None]        # (J,nt_loc) n . y_j
        nyj = np.zeros(ym.shape)
        nyj[:, tl] = ny[jj]
        in_pad = reg_t & (np.abs(xm) <= self.pad_xh)
        ymax_p = np.where(in_pad, ym, -np.inf).max(1)
        ymin_p = np.where(in_pad, ym, np.inf).min(1)
        band_p = in_pad & (ym >= (ymax_p - cfg.band)[:, None])
        band_n = in_pad & (ym <= (ymin_p + cfg.band)[:, None])
        ramp_p = np.clip((nyj - self.c2phi) / (self.cphi - self.c2phi), 0.0, 1.0)
        ramp_n = np.clip((-nyj - self.c2phi) / (self.cphi - self.c2phi), 0.0, 1.0)
        q_p = (ramp_p * band_p).sum(1) / np.maximum(band_p.sum(1), 1)
        q_n = (ramp_n * band_n).sum(1) / np.maximum(band_n.sum(1), 1)
        seen_p = np.any(band_p & (nyj > cfg.facing_cos), axis=1)
        seen_n = np.any(band_n & (-nyj > cfg.facing_cos), axis=1)
        q_p = np.where(seen_p, q_p, np.maximum(q_p, cfg.unobserved_credit))
        q_n = np.where(seen_n, q_n, np.maximum(q_n, cfg.unobserved_credit))
        antip = np.sqrt(q_p * q_n)

        # one side unseen: open fully and shift the TCP towards that side
        clr = 0.5 * (cfg.width_pad + cfg.open_extra)
        both = seen_p == seen_n
        w_cmd = np.where(both, np.clip(w_req + cfg.open_extra, cfg.min_width, cfg.max_width), cfg.max_width)
        c_shift = np.where(both, 0.5 * (ymin + ymax),
                           np.where(seen_n, ymin - clr + 0.5 * w_cmd,     # + side unseen: open towards it
                                    ymax + clr - 0.5 * w_cmd))            # - side unseen
        c_shift = np.clip(c_shift, -cfg.max_shift, cfg.max_shift)

        # final pose: hand shifted along y, commanded width, boxes swept over the approach
        Mn = len(w_cmd)
        lo2, hi2 = geom.boxes(w_cmd, cfg.include_wrist)          # (M,B,3) centred on the final TCP
        ys = ym - c_shift[:, None]                               # closing coord. w.r.t. final TCP
        hit = np.zeros(ym.shape, dtype=bool)
        for b in range(lo2.shape[1]):
            hit |= ((xm >= lo2[:, b, 0, None] - mi) & (xm <= hi2[:, b, 0, None] + mi)
                    & (ys >= lo2[:, b, 1, None] - mi) & (ys <= hi2[:, b, 1, None] + mi)
                    & (zrel >= lo2[:, b, 2, None] - cfg.approach_dist - mi) & (zrel <= hi2[:, b, 2, None] + mi))
        free = ~np.any(hit, axis=1)
        n_fin = (reg_t & (np.abs(ys) < 0.5 * w_cmd[:, None])).sum(1)
        free &= n_fin >= cfg.min_points
        # recheck obstacles between the fingers: the shift can bring a neighbour between the pads
        between = ((~is_t)[None] & (np.abs(xm) <= self.cx_half) & (np.abs(ys) < 0.5 * w_cmd[:, None])
                   & (zrel >= self.fz_lo) & (zrel <= self.fz_hi))
        free &= between.sum(1) <= cfg.max_obstacle_in_region
        if not np.any(free):
            return None

        y_ax = cs[jj] * e1[None] + sn[jj] * e2[None]            # (M,3)
        x_ax = sn[jj] * e1[None] - cs[jj] * e2[None]
        R = np.stack([x_ax, y_ax, np.broadcast_to(a, x_ax.shape)], axis=2)
        p = s[None] + dsel[:, None] * a[None] + c_shift[:, None] * y_ax
        lz = np.full(Mn, np.inf)
        for wv in np.unique(w_cmd):
            sel = w_cmd == wv
            Cw = geom.corners(wv, cfg.include_wrist)
            lz[sel] = (np.einsum("mk,nk->mn", R[sel, 2, :], Cw) + p[sel, 2, None]).min(1)
        rxy = p[:, :2] - np.asarray(cfg.base_xy)[None]
        rxy /= np.maximum(np.linalg.norm(rxy, axis=1, keepdims=True), 1e-9)
        toward = rxy @ a[:2]
        keep = free & (lz >= cfg.table_z + cfg.table_clear - 1e-9) & (toward >= -cfg.max_toward_base)
        if not np.any(keep):
            return None
        stats["n_final"] += int(keep.sum())

        # gravity lever arm of the target centre about the closing line: |(c-p).(y x e_z)|
        g_arm = np.stack([y_ax[:, 1], -y_ax[:, 0], np.zeros(Mn)], axis=1)   # y x e_z
        rho = np.abs(np.sum((self.center[None] - p) * g_arm, axis=1))
        q_center = np.exp(-0.5 * (rho / cfg.center_sigma) ** 2)
        q_count = np.minimum(1.0, n_fin / cfg.count_full)
        q_vert = np.full(Mn, np.clip(-a[2], 0.0, 1.0))
        score = (cfg.w_antipodal * antip + cfg.w_center * q_center
                 + cfg.w_count * q_count + cfg.w_vertical * q_vert)
        k = keep
        return dict(R=R[k], p=p[k], w=w_cmd[k], score=score[k], antipodal=antip[k], q_pos=q_p[k],
                    q_neg=q_n[k], one_sided=(~both[k]).astype(float), center=q_center[k],
                    center_dist=rho[k], n_region=n_fin[k].astype(float), count=q_count[k],
                    vertical=q_vert[k], depth=dsel[k], w_req=w_req[k], y_shift=c_shift[k],
                    tilt=np.full(k.sum(), float(beta)), approach_z=np.full(k.sum(), a[2]),
                    lowest_z=lz[k], psi=psis[jj[k]], origin=np.repeat(s[None], k.sum(), 0))


def propose_antipodal(target_pts, target_normals, obstacle_pts, rng=None, cfg=None,
                      geom: GripperGeometry | None = None, return_stats: bool = False):
    """Antipodal grasps on a partial target cloud, best first (at most cfg.max_grasps).

    target_pts: (N,3) world. target_normals: (N,3) outward unit normals, or None to estimate by PCA.
    obstacle_pts: (M,3) everything else, or None; the table is handled by cfg.table_z.
    With cfg.verify, every grasp keeps cfg.margin (m) from all input points over the last approach_dist.
    Returns: list[Grasp], plus a dict of counters if return_stats.
    """
    cfg = make_cfg(AntipodalConfig, cfg)
    geom_exact = geom or get_geometry()
    geom = geom_exact.symmetrized()                     # used by the search (flip-invariant)
    rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
    stats = dict(n_lines=0, n_frames=0, n_depth=0, n_region=0, n_width=0, n_final=0,
                 n_verified=0, n_verify_fail=0, n_out=0)

    TP = np.asarray(target_pts, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(TP), axis=1)
    TP = TP[finite]
    if target_normals is None:
        TN = estimate_normals_pca(TP, cfg.normal_k) if len(TP) >= 3 else np.zeros_like(TP)
    else:
        TN = np.asarray(target_normals, dtype=np.float64).reshape(-1, 3)[finite]
    nlen = np.linalg.norm(TN, axis=1)
    valid_n = np.isfinite(nlen) & (nlen > 1e-6)
    TN = np.where(valid_n[:, None], TN / np.where(valid_n, nlen, 1.0)[:, None], 0.0)
    if valid_n.sum() < cfg.min_points:
        return ([], stats) if return_stats else []
    lo_t, hi_t = TP.min(0), TP.max(0)
    center = 0.5 * (lo_t + hi_t)                        # bbox centre ~ centre of mass proxy
    it, rt = voxel_downsample(TP, cfg.voxel_target)
    tp, tn = TP[it], TN[it]

    # crop obstacles to what any output gripper (inflated, swept) can reach; enough for the exact check too
    blo, bhi = geom.boxes(max(cfg.max_width, geom.max_width), cfg.include_wrist)
    blo, bhi = blo - cfg.margin, bhi + cfg.margin
    blo[:, 2] -= cfg.approach_dist
    reach = (boxes_reach(blo, bhi) + max(cfg.max_depth, -cfg.min_depth) + cfg.max_shift
             + max((abs(x) for x in cfg.refine_shifts), default=0.0) + 1e-6)
    if obstacle_pts is not None and len(obstacle_pts):
        OP = np.asarray(obstacle_pts, dtype=np.float64).reshape(-1, 3)
        OP = OP[np.all(np.isfinite(OP), axis=1)]
        OP = OP[np.all((OP >= lo_t - reach) & (OP <= hi_t + reach), axis=1)]
        io, ro = voxel_downsample(OP, cfg.voxel_obstacle)
        op = OP[io]
    else:
        OP = np.zeros((0, 3))
        op, ro = OP, np.zeros(0)
    C = np.concatenate([tp, op])
    infl = cfg.margin + cfg.search_inflation * np.concatenate([rt, ro])   # per-point box inflation
    nt = len(tp)

    samplable = np.flatnonzero(np.linalg.norm(tn, axis=1) > 0.5)
    K = min(cfg.n_samples, len(samplable))
    samp = np.sort(rng.choice(samplable, size=K, replace=False))
    down = np.array([0.0, 0.0, -1.0])
    cos_dup = np.cos(np.deg2rad(5.0))
    lines = []                                          # (sample idx, approach, tilt)
    for k in samp:
        u0 = -tn[k]
        if u0 @ down < -0.95:                           # normal points down: bottom face
            continue
        kept = []
        for beta in cfg.tilts:
            a = _slerp(u0, down, beta)
            a /= np.linalg.norm(a)
            if a[2] > cfg.max_approach_z or any(a @ b > cos_dup for b in kept):
                continue
            kept.append(a)
            lines.append((k, a, beta))
    stats["n_lines"] = len(lines)
    psi_off = rng.uniform(0.0, np.pi / cfg.n_rot, size=len(lines)) if cfg.random_rot_offset \
        else np.zeros(len(lines))

    ctx = _LineEvaluator(C, infl, nt, tn, center, cfg, geom)
    recs = [ctx.eval_line(tp[k], a, ctx.psis + psi_off[li], beta, stats)
            for li, (k, a, beta) in enumerate(lines)]
    recs = [r for r in recs if r is not None]
    if not recs:
        return ([], stats) if return_stats else []
    cand = {key: np.concatenate([r[key] for r in recs]) for key in recs[0]}

    # sampling is coarse: shift the best frames along hand x and turn them half a step to centre the pads
    if cfg.refine_top > 0:
        order = np.argsort(-cand["score"], kind="stable")
        seeds = _diverse(cand, order, cfg.refine_top, cfg.dedup_pos * 2, np.deg2rad(2 * cfg.dedup_ang_deg))
        dpsi = 0.5 * np.pi / cfg.n_rot
        more = []
        for i in seeds:
            x_ax = cand["R"][i, :, 0]
            for dx in cfg.refine_shifts:
                r = ctx.eval_line(cand["origin"][i] + dx * x_ax, cand["R"][i, :, 2],
                                  cand["psi"][i] + np.array([-dpsi, 0.0, dpsi]), cand["tilt"][i], stats)
                if r is not None:
                    more.append(r)
        if more:
            cand = {key: np.concatenate([cand[key]] + [r[key] for r in more]) for key in cand}

    # sort, deduplicate, verify exactly; repeat so a suppressed neighbour can take a failed slot
    order = np.argsort(-cand["score"], kind="stable")[: max(8 * cfg.max_grasps, 256)]
    R0 = cand["R"][order]
    Ts = canonicalize_about_approach(make_pose(R0, cand["p"][order]), cfg.base_xy)
    # canonicalisation may flip y; keep y_shift meaning TCP = s + depth*a + y_shift*closing
    y_sign = np.where(np.sum(Ts[:, :3, 1] * R0[:, :, 1], axis=1) < 0, -1.0, 1.0)
    ws = cand["w"][order]
    grasps = []
    for n, i in enumerate(order):
        info = {k: float(cand[k][i]) for k in _INFO_KEYS}
        info["y_shift"] *= y_sign[n]
        grasps.append(Grasp(T=Ts[n], width=float(ws[n]), score=float(cand["score"][i]),
                            method=cfg.method, info=info))
    slot = {id(g): n for n, g in enumerate(grasps)}
    status = np.zeros(len(grasps), dtype=np.int8)       # 0 unverified, 1 verified, -1 collides
    collider = CloudCollider(np.concatenate([TP, OP]), cfg.verify_voxel) if cfg.verify else None
    obs_lab = voxel_labels(OP, cfg.voxel_obstacle) if cfg.verify else None
    while True:
        alive = [g for n, g in enumerate(grasps) if status[n] >= 0]
        out = dedup_grasps(alive, cfg.dedup_pos, np.deg2rad(cfg.dedup_ang_deg))[: cfg.max_grasps]
        if collider is None:
            break
        todo = np.array([slot[id(g)] for g in out if status[slot[id(g)]] == 0], dtype=np.int64)
        if todo.size == 0:
            break
        bad = collider.collides(Ts[todo], ws[todo], cfg.margin, geom_exact, cfg.include_wrist,
                                sweep=cfg.approach_dist)
        # obstacles between the fingers, counted in search-grid voxels so the tolerance means the same
        bad |= closing_volume_voxel_counts(Ts[todo], ws[todo], OP, obs_lab, geom_exact) > cfg.max_obstacle_in_region
        status[todo] = np.where(bad, -1, 1)
        stats["n_verified"] += int(todo.size)
        stats["n_verify_fail"] += int(bad.sum())
    stats["n_out"] = len(out)
    return (out, stats) if return_stats else out
