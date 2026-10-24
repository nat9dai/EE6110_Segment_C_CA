"""CBF-QP safety filter: the joint velocity closest to the nominal one that keeps every robot
sphere outside each barrier (distance field or table), with optional soft (slack) barriers.
Barrier value h = d(p_i) - r_i - margin [m]; h < 0 means inside the margin."""
from __future__ import annotations

import time
from ctypes import c_int
from dataclasses import dataclass

import numpy as np

try:
    import daqp as _daqp
except Exception:  # pragma: no cover - daqp is in the Isaac Sim python
    _daqp = None

__all__ = ["Barrier", "CBFFilter", "SphereSet", "robot_sphere_set", "barrier_values",
           "barrier_indices", "TABLE"]

TABLE = "table"
_INF = 1e30          # DAQP's "no bound"


@dataclass
class Barrier:
    """One barrier applied to a set of spheres.

    field: has query(pts (N,3)) -> (d (N,), grad (N,3)), negative inside; or 'table' (plane z = 0).
    margin: delta [m]; None -> CBFFilter.table_margin for 'table', else 0.
    sphere_mask: robot spheres (bool (S,) or indices), None = all; name '' -> 'table' or 'b<k>'.
    """
    field: object
    margin: float | None = None
    soft: bool = False
    name: str = ""
    sphere_mask: np.ndarray | None = None
    include_extra: bool = True
    alpha: float | None = None

    @property
    def is_table(self) -> bool:
        return isinstance(self.field, str) and self.field == TABLE


def _barrier_name(b: Barrier, k: int) -> str:
    if b.name:
        return b.name
    return TABLE if b.is_table else f"b{k}"


def barrier_values(b: Barrier, centers: np.ndarray, radii: np.ndarray,
                   table_margin: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """h (N,) [m] and grad d (N,3) of barrier ``b`` at the spheres; h = d(c) - r - margin.
    Non-finite distances give h = +inf and zero gradient (no constraint)."""
    n = len(radii)
    if b.is_table:
        margin = table_margin if b.margin is None else float(b.margin)
        grad = np.zeros((n, 3))
        grad[:, 2] = 1.0
        h = centers[:, 2] - radii - margin
        bad = ~np.isfinite(h)
        if bad.any():                      # NaN would hide in min()
            h = np.where(bad, np.inf, h)
            grad[bad] = 0.0
        return h, grad
    if isinstance(b.field, str):
        raise ValueError(f"unknown barrier field {b.field!r} (only 'table' is built in)")
    margin = 0.0 if b.margin is None else float(b.margin)
    if n == 0:
        return np.zeros(0), np.zeros((0, 3))
    d, grad = b.field.query(centers)
    d = np.asarray(d, dtype=float).reshape(n)
    grad = np.asarray(grad, dtype=float).reshape(n, 3)
    bad = ~np.isfinite(d)
    if bad.any():
        d = np.where(bad, np.inf, d)
        grad = np.where(bad[:, None], 0.0, grad)
    return d - radii - margin, grad


@dataclass
class SphereSet:
    """Robot spheres followed by extra spheres: centres (N,3), radii (N,), jac (N,3,7)."""
    centers: np.ndarray
    radii: np.ndarray
    jac: np.ndarray
    n_robot: int
    exclude: np.ndarray        # bool (n_robot,) robot spheres removed from all barriers

    @property
    def n_extra(self) -> int:
        return len(self.radii) - self.n_robot


def _as_mask(sel, n: int) -> np.ndarray:
    """None -> all True; bool mask or index array -> bool mask of length n."""
    if sel is None:
        return np.ones(n, dtype=bool)
    sel = np.asarray(sel)
    if sel.dtype == bool:
        if sel.shape != (n,):
            raise ValueError(f"sphere mask has shape {sel.shape}, expected ({n},)")
        return sel
    m = np.zeros(n, dtype=bool)
    m[sel.astype(np.int64)] = True
    return m


def robot_sphere_set(panda, q, width, extra_spheres=None, exclude_spheres=None) -> SphereSet:
    """Robot collision spheres at (q, width) stacked with optional extra spheres."""
    c, r, J, _ = panda.spheres(q, width)
    S = len(r)
    exclude = ~_as_mask(None, S) if exclude_spheres is None else _as_mask(exclude_spheres, S)
    if extra_spheres is not None and len(extra_spheres[1]) > 0:
        ec, er, eJ = extra_spheres
        c = np.concatenate([c, np.asarray(ec, float).reshape(-1, 3)])
        r = np.concatenate([r, np.asarray(er, float).reshape(-1)])
        J = np.concatenate([J, np.asarray(eJ, float).reshape(-1, 3, 7)])
    return SphereSet(c, r, J, S, exclude)


def barrier_indices(b: Barrier, sph: SphereSet) -> np.ndarray:
    """Indices into ``sph`` of the spheres barrier ``b`` applies to."""
    m = _as_mask(b.sphere_mask, sph.n_robot) & ~sph.exclude
    idx = np.flatnonzero(m)
    if b.include_extra and sph.n_extra > 0:
        idx = np.concatenate([idx, sph.n_robot + np.arange(sph.n_extra)])
    return idx


class CBFFilter:
    """CBF-QP safety filter on joint velocities.

    alpha, alpha_q: class-K gains for sphere and joint-limit barriers [1/s].
    table_margin [m]; activation_dist [m]: only rows with h < activation_dist enter the QP.
    joint_buffer [rad]: gap kept from the joint limits; reg: Tikhonov weight on |qd|^2.
    relax_weight: per-row slack penalty on hard rows when the QP is infeasible.
    """

    def __init__(self, panda, alpha=5.0, alpha_q=5.0, table_margin=0.01, use_joint_limits=True,
                 slack_weight=1e4, *, activation_dist=0.12, reg=1e-4, joint_buffer=0.02,
                 relax_weight=1e7, slack_per_row=True):
        self.panda = panda
        self.alpha = float(alpha)
        self.alpha_q = float(alpha_q)
        self.table_margin = float(table_margin)
        self.use_joint_limits = bool(use_joint_limits)
        self.slack_weight = float(slack_weight)
        self.activation_dist = float(activation_dist)
        self.reg = float(reg)
        self.joint_buffer = float(joint_buffer)
        self.relax_weight = float(relax_weight)
        self.slack_per_row = bool(slack_per_row)

    def velocity_bounds(self, q) -> tuple[np.ndarray, np.ndarray]:
        """Box on qd from velocity limits and the joint-position CBF (lb <= ub guaranteed)."""
        qd_max = self.panda.qd_max
        lb, ub = -qd_max.copy(), qd_max.copy()
        if self.use_joint_limits:
            buf = self.joint_buffer
            lb = np.maximum(lb, -self.alpha_q * (q - self.panda.q_lower - buf))
            ub = np.minimum(ub, self.alpha_q * (self.panda.q_upper - buf - q))
            lb = np.minimum(lb, ub)                  # only if q is far outside the buffer
        return lb, ub

    def build_rows(self, sph: SphereSet, barriers):
        """Evaluate all barriers and build the active CBF rows A qd >= b.

        Returns (rows, h_min, n_dead): per-barrier dicts {name, soft, A (n,7), b (n,), sphere, h},
        name -> min h over all its spheres, and the count of violated rows with zero A."""
        rows, h_min, n_dead = [], {}, 0
        for k, bar in enumerate(barriers):
            name = _barrier_name(bar, k)
            idx = barrier_indices(bar, sph)
            if len(idx) == 0:
                h_min[name] = min(h_min.get(name, np.inf), np.inf)
                continue
            h, g = barrier_values(bar, sph.centers[idx], sph.radii[idx], self.table_margin)
            h_min[name] = min(h_min.get(name, np.inf), float(np.min(h)))
            act = h < self.activation_dist
            if not act.any():
                continue
            ia, h, g = idx[act], h[act], g[act]
            A = np.einsum("ni,nij->nj", g, sph.jac[ia])                  # grad d^T J_i   (n,7)
            alpha = self.alpha if bar.alpha is None else float(bar.alpha)
            b = -alpha * h
            # a row with ~zero A cannot be changed by qd: drop it
            live = np.einsum("nj,nj->n", A, A) > 1e-16
            n_dead += int(np.sum(~live & (b > 0)))                      # uncontrollable violation
            if not live.all():
                A, b, ia, h = A[live], b[live], ia[live], h[live]
            if len(b):
                rows.append(dict(name=name, soft=bool(bar.soft), A=A, b=b, sphere=ia, h=h))
        return rows, h_min, n_dead

    def _solve(self, qd_nom, lb, ub, A, b, slack_cols, slack_w):
        """QP: min |qd - qd_nom|^2 + reg|qd|^2 + sum_k w_k s_k^2 s.t. A qd + E s >= b, s >= 0,
        lb <= qd <= ub; slack_cols[i] is row i's slack column (-1 = none). Returns (x, ok, name)."""
        m, ns = len(b), len(slack_w)
        n = 7 + ns
        H = np.empty(n)
        H[:7] = 2.0 * (1.0 + self.reg)
        H[7:] = 2.0 * np.asarray(slack_w)
        f = np.zeros(n)
        f[:7] = -2.0 * qd_nom
        G = np.zeros((m, n))
        G[:, :7] = A
        if ns:
            r = np.flatnonzero(slack_cols >= 0)
            G[r, 7 + slack_cols[r]] = 1.0
        lo = np.concatenate([lb, np.zeros(ns)])
        hi = np.concatenate([ub, np.full(ns, _INF)])

        if _daqp is not None:
            try:
                x, _, flag, _ = _daqp.solve(np.diag(H), f, G, np.concatenate([hi, np.full(m, _INF)]),
                                            np.concatenate([lo, b]), np.zeros(n + m, dtype=c_int))
                if flag > 0:
                    return np.asarray(x), True, "daqp"
                if flag == -1:                        # certified infeasible
                    return None, False, "daqp"
            except Exception:
                pass
        try:
            import qpsolvers
            x = qpsolvers.solve_qp(np.diag(H), f, -G, -b, lb=lo, ub=np.minimum(hi, 1e20),
                                   solver="quadprog")
            if x is not None and np.all(np.isfinite(x)):
                return np.asarray(x), True, "quadprog"
        except Exception:
            pass
        return None, False, "none"

    def filter(self, q, qd_nom, width, barriers, exclude_spheres=None, extra_spheres=None):
        """Filter a nominal joint velocity; returns (qd (7,), info).

        q, qd_nom: (7,); width: gripper opening [m].
        extra_spheres: (centres (E,3), radii (E,), jac (E,3,7)), e.g. a grasped object.
        info: h_min {name: min h before the step}, n_rows, n_active, n_dead, slack {name: max},
        status 'ok' (nominal feasible, no QP) | 'optimal' | 'relaxed' | 'failed', timings,
        intervention |qd - qd_nom|, nominal_ok (False if a non-finite qd_nom became zero).
        """
        t0 = time.perf_counter()
        q = np.asarray(q, dtype=float).reshape(7)
        qd_nom = np.asarray(qd_nom, dtype=float).reshape(7)
        barriers = [] if barriers is None else list(barriers)
        slack = {_barrier_name(b, k): 0.0 for k, b in enumerate(barriers) if b.soft}
        nominal_ok = bool(np.all(np.isfinite(qd_nom)))
        info = dict(h_min={}, n_rows=0, n_rows_per={}, n_active=0, n_dead=0, slack=slack,
                    status="ok", solver="none", solve_ms=0.0, total_ms=0.0, intervention=0.0,
                    nominal_ok=nominal_ok)
        if not (np.all(np.isfinite(q)) and np.isfinite(float(width))):
            # invalid state: stop rather than emit NaN
            info.update(h_min={_barrier_name(b, k): float("nan") for k, b in enumerate(barriers)},
                        status="failed", intervention=float(np.linalg.norm(qd_nom)) if nominal_ok
                        else float("nan"), total_ms=1e3 * (time.perf_counter() - t0))
            return np.zeros(7), info
        if not nominal_ok:
            qd_nom = np.zeros(7)                    # filter a stop command instead of NaN / inf
        sph = robot_sphere_set(self.panda, q, width, extra_spheres, exclude_spheres)
        rows, h_min, n_dead = self.build_rows(sph, barriers)
        lb, ub = self.velocity_bounds(q)
        info.update(h_min=h_min, n_dead=n_dead)
        if rows:
            A = np.concatenate([r["A"] for r in rows])
            b = np.concatenate([r["b"] for r in rows])
            soft_row = np.concatenate([np.full(len(r["b"]), r["soft"]) for r in rows])
            for r in rows:
                info["n_rows_per"][r["name"]] = info["n_rows_per"].get(r["name"], 0) + len(r["b"])
        else:
            A, b, soft_row = np.zeros((0, 7)), np.zeros(0), np.zeros(0, bool)
        info["n_rows"] = len(b)

        # fast path: nominal feasible, or only the box binds so clipping is exact
        tol = 1e-9
        rows_ok = np.all(A @ qd_nom >= b - tol) if len(b) else True
        if rows_ok and np.all(qd_nom >= lb - tol) and np.all(qd_nom <= ub + tol):
            qd = qd_nom.copy()
            info["total_ms"] = 1e3 * (time.perf_counter() - t0)
            return qd, info
        if len(b) == 0:
            qd = np.clip(qd_nom, lb, ub)
            info.update(status="optimal", intervention=float(np.linalg.norm(qd - qd_nom)),
                        total_ms=1e3 * (time.perf_counter() - t0))
            return qd, info

        row_names = np.array([str(r["name"]) for r in rows for _ in range(len(r["b"]))], dtype=object)
        slack_cols = np.full(len(b), -1, dtype=np.int64)
        col_names, slack_w = [], []
        if self.slack_per_row:
            soft_idx = np.flatnonzero(soft_row)
            slack_cols[soft_idx] = np.arange(len(soft_idx))
            col_names = list(row_names[soft_idx])
            slack_w = [self.slack_weight] * len(soft_idx)
        else:
            for nm in dict.fromkeys(row_names[soft_row]):
                slack_cols[(row_names == nm) & soft_row] = len(col_names)
                col_names.append(nm); slack_w.append(self.slack_weight)

        ts = time.perf_counter()
        x, ok, solver = self._solve(qd_nom, lb, ub, A, b, slack_cols, slack_w)
        status = "optimal"
        row_col = slack_cols                         # slack column of each row (-1 = none)
        if not ok:
            # infeasible: violated hard rows keep A qd >= 0 hard and slack only their recovery
            # (qd = 0 is then feasible inside the joint box); failing that, slack every hard row.
            status = "relaxed"
            hard = ~soft_row
            viol = hard & (b > 0)
            nv = int(viol.sum())
            if nv:
                vcols = len(col_names) + np.arange(nv)
                x, ok, solver = self._solve(
                    qd_nom, lb, ub, np.vstack([A, A[viol]]),
                    np.concatenate([np.where(viol, 0.0, b), b[viol]]),
                    np.concatenate([slack_cols, vcols]),
                    list(slack_w) + [self.relax_weight] * nv)
                if ok:
                    row_col = slack_cols.copy()
                    row_col[viol] = vcols
                    col_names = list(col_names) + list(row_names[viol])
            if not ok:
                hidx = np.flatnonzero(hard)
                cols2 = slack_cols.copy()
                cols2[hidx] = len(col_names) + np.arange(len(hidx))
                x, ok, solver = self._solve(qd_nom, lb, ub, A, b, cols2,
                                            list(slack_w) + [self.relax_weight] * len(hidx))
                if ok:
                    row_col = cols2
                    col_names = list(col_names) + list(row_names[hidx])
        info["solve_ms"] = 1e3 * (time.perf_counter() - ts)
        info["solver"] = solver

        if not ok:
            qd = np.clip(np.zeros(7), lb, ub)       # stop (inside the joint-limit box)
            info["status"] = "failed"
        else:
            qd = np.clip(x[:7], lb, ub)
            s = np.maximum(x[7:], 0.0)
            for j, nm in enumerate(col_names):
                info["slack"][nm] = max(info["slack"].get(nm, 0.0), float(s[j]))
            s_row = np.zeros(len(b))
            has = row_col >= 0
            s_row[has] = s[row_col[has]]
            resid = A @ qd - b + s_row
            info["n_active"] = int(np.sum(resid < 1e-6))
            info["status"] = status
        info["intervention"] = float(np.linalg.norm(qd - qd_nom))
        info["total_ms"] = 1e3 * (time.perf_counter() - t0)
        return qd, info

    __call__ = filter

    def evaluate(self, q, width, barriers, exclude_spheres=None, extra_spheres=None) -> dict:
        """Min h per barrier without the QP: {name: h [m]}; all nan if q or width is non-finite."""
        q = np.asarray(q, dtype=float)
        if not (np.all(np.isfinite(q)) and np.isfinite(float(width))):
            return {_barrier_name(b, k): float("nan") for k, b in enumerate(barriers)}
        sph = robot_sphere_set(self.panda, q, width, extra_spheres, exclude_spheres)
        out = {}
        for k, bar in enumerate(barriers):
            name = _barrier_name(bar, k)
            idx = barrier_indices(bar, sph)
            if len(idx) == 0:
                out.setdefault(name, np.inf)
                continue
            h, _ = barrier_values(bar, sph.centers[idx], sph.radii[idx], self.table_margin)
            out[name] = min(out.get(name, np.inf), float(np.min(h)))
        return out
