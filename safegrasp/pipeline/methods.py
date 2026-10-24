"""Method registry: grasp proposer x executor (safety layer) x perception views."""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Method:
    name: str
    proposer: str                 # 'topdown' | 'antipodal' | 'safe'
    executor: str                 # 'nominal' | 'apf' | 'cbf' | 'oacbf'
    views: tuple = ("main",)
    alpha: float = 5.0            # CBF class-K gain [1/s]
    margin_obs: float = 0.01      # δ_obs [m]
    margin_unk: float = 0.005     # δ_unk [m]
    margin_table: float = 0.01    # δ_tab [m], arm + hand spheres only
    unk_soft: bool = True         # unknown-space barrier has slack
    carve_radius: float = 0.02    # [m] unknown voxels this close to the target count as target
    noise: float = 1.0            # depth-noise multiplier (0 = perfect depth)
    recover: str = "none"         # on-deadlock recovery: 'none' | 'reobserve' | 'relax'
    notes: str = ""

    def with_(self, **kw) -> "Method":
        return replace(self, **kw)


BASE = {
    "td_nom": Method("td_nom", "topdown", "nominal"),
    "ant_nom": Method("ant_nom", "antipodal", "nominal"),
    "ant_apf": Method("ant_apf", "antipodal", "apf"),
    "ant_cbf": Method("ant_cbf", "antipodal", "cbf"),
    "ant_oacbf": Method("ant_oacbf", "antipodal", "oacbf"),
    "safe_nom": Method("safe_nom", "safe", "nominal"),
    "safe_cbf": Method("safe_cbf", "safe", "cbf"),
    "safe_oacbf": Method("safe_oacbf", "safe", "oacbf"),
}


def get_method(spec: str) -> Method:
    """Parse ``name`` or ``name@key=val,...``, e.g. ``safe_oacbf@alpha=2`` or ``ant_cbf@views=main+left+right``."""
    if "@" not in spec:
        return BASE[spec]
    base, kv = spec.split("@", 1)
    m = BASE[base]
    kw = {}
    for item in kv.split(","):
        k, v = item.split("=")
        if k == "views":
            kw[k] = tuple(v.split("+"))
        elif k in ("unk_soft",):
            kw[k] = v.lower() in ("1", "true", "yes")
        elif k in ("proposer", "executor", "notes", "recover"):
            kw[k] = v
        else:
            kw[k] = float(v)
    from safegrasp.pipeline.recovery import RECOVER_MODES
    if kw.get("recover", "none") not in RECOVER_MODES:  # fail at launch, not inside every trial
        raise ValueError(f"unknown recover mode {kw['recover']!r} in {spec!r}")
    return m.with_(name=spec, **kw)
