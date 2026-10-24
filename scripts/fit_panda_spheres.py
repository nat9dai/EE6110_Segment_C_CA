"""Fit collision spheres to the Isaac Lab Franka collision hulls into safegrasp/kinematics/panda_spheres.json.

    ~/isaacsim/python.sh scripts/fit_panda_spheres.py [--collision FILE.npz] [--restarts 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import trimesh
from scipy.optimize import minimize
from sklearn.cluster import KMeans

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "safegrasp", "kinematics", "panda_spheres.json")
ALL_LINKS = ["panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5",
             "panda_link6", "panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger"]
# spheres per link; link0 is the fixed base and gets none
ALLOC = {"panda_link1": 5, "panda_link2": 5, "panda_link3": 6, "panda_link4": 6, "panda_link5": 8,
         "panda_link6": 6, "panda_link7": 5, "panda_hand": 8, "panda_leftfinger": 3, "panda_rightfinger": 3}
E_FIT = 0.004        # m, how far the hull may stick out of the spheres (limit 5 mm)


def halfspaces(h: trimesh.Trimesh):
    """Hull as {x : n_i . x <= b_i}."""
    n = h.face_normals
    return n, np.einsum("ij,ij->i", n, h.triangles[:, 0, :])


def surface_samples(h: trimesh.Trimesh, n: int, seed: int) -> np.ndarray:
    """Sample the hull surface, its vertices and its edges, so corners are always covered."""
    pts, _ = trimesh.sample.sample_surface_even(h, n, seed=seed)
    e = h.vertices[h.edges_unique]
    t = np.linspace(0.0, 1.0, 6)
    edge_pts = (e[:, 0, None, :] * t[None, :, None] + e[:, 1, None, :] * (1 - t)[None, :, None]).reshape(-1, 3)
    return np.vstack([pts, h.vertices, edge_pts])


def coverage_gap(P, c, r):
    """g_p = min_k ||p - c_k|| - r_k  (<= 0: p inside the union) and the arg-min sphere."""
    d = np.linalg.norm(P[:, None, :] - c[None], axis=-1) - r[None]
    return d.min(1), d.argmin(1)


def excess_volume(h, c, r, n=200000, seed=0):
    """Monte-Carlo volumes: (union \\ hull, union, hull \\ union)  [m^3]."""
    rng = np.random.default_rng(seed)
    lo = np.minimum(h.bounds[0], (c - r[:, None]).min(0))
    hi = np.maximum(h.bounds[1], (c + r[:, None]).max(0))
    X = rng.uniform(lo, hi, size=(n, 3))
    V = np.prod(hi - lo)
    nrm, b = halfspaces(h)
    in_hull = (X @ nrm.T - b[None] <= 0).all(1)
    in_u = (np.linalg.norm(X[:, None] - c[None], axis=-1) <= r[None]).any(1)
    return V * (in_u & ~in_hull).mean(), V * in_u.mean(), V * (in_hull & ~in_u).mean()


def fit_link(h, K, e_fit=E_FIT, seed=0, n_surf=1500):
    """Fit K spheres covering hull h within e_fit; return centres (K, 3) and radii (K,) in m."""
    rng = np.random.default_rng(seed)
    P = surface_samples(h, n_surf, seed)
    nrm, b = halfspaces(h)
    # start from k-means centres of interior samples, radii covering each cluster
    X = rng.uniform(h.bounds[0], h.bounds[1], size=(20000, 3))
    X = X[(X @ nrm.T - b[None] <= -0.002).all(1)]
    c0 = KMeans(K, n_init=4, random_state=seed).fit(X).cluster_centers_
    g, a = coverage_gap(P, c0, np.zeros(K))
    r0 = np.array([g[a == k].max() if np.any(a == k) else 0.01 for k in range(K)])
    # sphere depth at exterior samples (times volume per sample) estimates the excess volume
    pad = 0.06
    Q = rng.uniform(h.bounds[0] - pad, h.bounds[1] + pad, size=(40000, 3))
    Q = Q[(Q @ nrm.T - b[None] > 0).any(1)]
    wq = 1e6 * np.prod(h.extents + 2 * pad) / 40000.0
    ar = np.arange(len(P))

    def f(x, mu):
        c, r = x[:3 * K].reshape(K, 3), x[3 * K:]
        D = P[:, None, :] - c[None]
        dist = np.linalg.norm(D, axis=-1)
        k = (dist - r[None]).argmin(1)
        viol = np.maximum(dist[ar, k] - r[k] - e_fit, 0.0)
        DQ = Q[:, None, :] - c[None]
        dq = np.linalg.norm(DQ, axis=-1)
        pen = np.maximum(r[None] - dq, 0.0)
        val = wq * pen.sum() + mu * np.sum(viol ** 2)
        act = (pen > 0) * wq
        gr = act.sum(0)
        gc = np.einsum("qk,qkd->kd", act / np.maximum(dq, 1e-9), DQ)
        idx = viol > 0
        w = 2 * mu * viol[idx] / np.maximum(dist[ar, k][idx], 1e-9)
        np.add.at(gc, k[idx], -w[:, None] * D[ar, k][idx])
        np.add.at(gr, k[idx], -2 * mu * viol[idx])
        return val, np.concatenate([gc.ravel(), gr])

    x = np.concatenate([c0.ravel(), r0])
    bounds = [(None, None)] * (3 * K) + [(0.003, None)] * K
    for mu in [1e4, 1e5, 1e6, 1e7, 1e8]:
        x = minimize(f, x, args=(mu,), jac=True, method="L-BFGS-B", bounds=bounds,
                     options=dict(maxiter=500)).x
    c, r = x[:3 * K].reshape(K, 3), x[3 * K:].copy()
    # grow radii until a fresh dense sample is covered within e_fit
    Pd = surface_samples(h, 20000, seed + 100)
    for _ in range(3):
        gmin, k = coverage_gap(Pd, c, r)
        bad = gmin > e_fit
        for kk in np.unique(k[bad]):
            r[kk] += gmin[bad & (k == kk)].max() - e_fit + 1e-5
    return c, r


def evaluate(h, c, r, seed=7) -> dict:
    Pt = surface_samples(h, 50000, seed)
    g, _ = coverage_gap(Pt, c, r)
    ev, uv, missed = excess_volume(h, c, r)
    nrm, b = halfspaces(h)
    sd = (b[None] - c @ nrm.T).min(1)        # depth of each centre inside the hull
    return dict(max_protrusion_mm=1e3 * max(float(g.max()), 0.0), hull_vol_cm3=h.volume * 1e6,
                excess_vol_cm3=ev * 1e6, union_vol_cm3=uv * 1e6, missed_vol_cm3=missed * 1e6,
                max_inflation_mm=1e3 * float((r - sd).max()), mean_r_mm=1e3 * float(r.mean()))


def best_fit(h, K, restarts=4):
    best = None
    for s in range(restarts):
        c, r = fit_link(h, K, seed=s)
        m = evaluate(h, c, r)
        if best is None or m["excess_vol_cm3"] < best[2]["excess_vol_cm3"]:
            best = (c, r, m)
    return best


def load_hulls(collision_npz: str | None) -> dict[str, trimesh.Trimesh]:
    hulls = {}
    if collision_npz:
        C = np.load(collision_npz)
        for link in ALL_LINKS:
            if f"{link}__v" in C:
                hulls[link] = trimesh.Trimesh(C[f"{link}__v"], C[f"{link}__f"], process=True).convex_hull
    else:
        with open(OUT_JSON) as f:
            data = json.load(f)
        for link, v in data["hulls"].items():
            hulls[link] = trimesh.convex.convex_hull(np.asarray(v, dtype=float))
    return hulls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collision", default=None, help="npz from verify_fk_isaac.py --dump-collision")
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    hulls = load_hulls(args.collision)

    links, metrics = {}, {}
    for link, K in ALLOC.items():
        if link == "panda_rightfinger":   # right mesh = left mesh rotated by pi about z
            c = links["panda_leftfinger"]["c"] * np.array([-1.0, -1.0, 1.0])
            r = links["panda_leftfinger"]["r"]
            m = evaluate(hulls[link], c, r)
        else:
            c, r, m = best_fit(hulls[link], K, args.restarts)
        links[link] = dict(c=c, r=r)
        metrics[link] = {k: round(float(v), 3) for k, v in m.items()}
        print(link, K, metrics[link], flush=True)
    total = sum(len(v["r"]) for v in links.values())
    print("total spheres", total)

    out = {
        "description": "Franka Panda collision spheres fitted to the Isaac Lab FRANKA_PANDA_HIGH_PD_CFG "
                       "collision meshes (convex hulls). Centres are in each link's body frame [m]; the "
                       "union of a link's spheres contains its collision hull up to max_protrusion_mm.",
        "frame": "link body frame (Isaac / URDF), metres",
        "fit_allowance_m": E_FIT,
        "n_spheres": int(total),
        "links": {k: {"centers": np.round(v["c"], 5).tolist(), "radii": np.round(v["r"] + 5e-6, 5).tolist()}
                  for k, v in links.items()},
        "metrics": metrics,
        # hull vertices (link frame) let tests and refits run without Isaac
        "hulls": {k: np.round(h.vertices, 6).tolist() for k, h in hulls.items()},
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
