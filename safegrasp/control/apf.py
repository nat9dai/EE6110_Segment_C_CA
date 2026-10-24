"""APF baseline: Khatib's repulsive push on each robot sphere, mapped to the joints by J^T and added
to the nominal command. Uses the same barriers as the CBF filter but gives no safety guarantee."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safegrasp.control.cbf import (Barrier, _barrier_name, barrier_indices, barrier_values,
                                   robot_sphere_set)

__all__ = ["APFParams", "apf_qd", "as_barriers"]


@dataclass
class APFParams:
    """APF gains: eta [m^3/s], influence distance d0 [m], clearance floor c_min [m],
    table_margin [m] for 'table' barriers with margin None, max_force cap on |F_i| [m/s]."""
    eta: float = 2e-5
    d0: float = 0.05
    c_min: float = 1e-3
    soft_gain: float = 1.0
    table_margin: float = 0.01
    max_force: float | None = None


def as_barriers(fields) -> list[Barrier]:
    """Accept a list of Barrier or a dict name -> (field, margin)."""
    if isinstance(fields, dict):
        return [Barrier(f, m, name=k) for k, (f, m) in fields.items()]
    return list(fields)


def apf_qd(panda, q, qd_nom, width, fields, params: APFParams | None = None, *,
           extra_spheres=None, exclude_spheres=None, return_info: bool = False):
    """APF-modified joint velocity (7,); return_info=True also returns min clearance per barrier.

    fields: list[Barrier] or dict name -> (field, margin).
    extra_spheres: (centres (E,3), radii (E,), jac (E,3,7)) of attached-object spheres."""
    p = APFParams() if params is None else params
    q = np.asarray(q, dtype=float).reshape(7)
    qd_nom = np.asarray(qd_nom, dtype=float).reshape(7)
    if not (np.all(np.isfinite(q)) and np.isfinite(float(width))):
        # invalid state: stop rather than emit NaN
        qd = np.zeros(7)
        return (qd, dict(h_min={}, n_repulsive=0, rep_norm=0.0, intervention=float("nan"),
                         valid=False)) if return_info else qd
    if not np.all(np.isfinite(qd_nom)):
        qd_nom = np.zeros(7)                      # invalid nominal -> repulsion acts on a stop
    sph = robot_sphere_set(panda, q, width, extra_spheres, exclude_spheres)
    qd_rep = np.zeros(7)
    h_min, n_in = {}, 0
    for k, bar in enumerate(as_barriers(fields)):
        name = _barrier_name(bar, k)
        idx = barrier_indices(bar, sph)
        if len(idx) == 0:
            continue
        h, g = barrier_values(bar, sph.centers[idx], sph.radii[idx], p.table_margin)
        h_min[name] = min(h_min.get(name, np.inf), float(np.min(h)))
        inside = h < p.d0
        if not inside.any():
            continue
        c = np.maximum(h[inside], p.c_min)
        mag = p.eta * (1.0 / c - 1.0 / p.d0) / c ** 2
        if p.max_force is not None:
            mag = np.minimum(mag, p.max_force)
        if bar.soft:
            mag = mag * p.soft_gain
        F = mag[:, None] * g[inside]                                     # (n,3)
        qd_rep += np.einsum("ni,nij->j", F, sph.jac[idx[inside]])        # sum_i J_i^T F_i
        n_in += int(inside.sum())
    qd = np.clip(qd_nom + qd_rep, -panda.qd_max, panda.qd_max)
    if return_info:
        return qd, dict(h_min=h_min, n_repulsive=n_in, rep_norm=float(np.linalg.norm(qd_rep)),
                        intervention=float(np.linalg.norm(qd - qd_nom)), valid=True)
    return qd
