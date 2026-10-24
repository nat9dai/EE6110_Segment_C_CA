"""On-deadlock recovery for the OA-CBF executor: on a transit or approach stall, act and resume.

'reobserve' re-runs perception with main+left+right views; 'relax' sets the unknown-space margin to 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

RECOVER_MODES = ("none", "reobserve", "relax")
RECOVER_PHASES = ("transit", "approach")


@dataclass
class RecoveryState:
    """Per-trial recovery bookkeeping: decides when to recover and logs each event."""
    mode: str = "none"
    executor: str = "oacbf"
    max_recoveries: int = 2
    events: list = field(default_factory=list)

    def __post_init__(self):
        if self.mode not in RECOVER_MODES:
            raise ValueError(f"unknown recover mode {self.mode!r}, expected one of {RECOVER_MODES}")

    @property
    def n(self) -> int:
        return len(self.events)

    def should_recover(self, status: str, phase: str) -> bool:
        """Return True if a phase that ended with ``status`` should recover instead of retreating."""
        return (self.mode != "none" and self.executor == "oacbf" and status == "stall"
                and phase in RECOVER_PHASES and self.n < self.max_recoveries)

    def record(self, phase: str, t: float, **info) -> dict:
        ev = dict(k=self.n + 1, mode=self.mode, phase=phase, t=float(t), **info)
        self.events.append(ev)
        return ev


def approach_resume_time(p_tcp, p_grasp, axis, L: float, T_dur: float) -> float:
    """Return the time on the approach reference that matches the TCP's current depth along ``axis``.

    Points are (3,) world frame [m], ``axis`` points pre-grasp -> grasp, L [m], T_dur [s].
    Example: TCP 4 cm past the pre-grasp, L = 12 cm, T_dur = 2 s -> 0.667 s.
    """
    p_pre = np.asarray(p_grasp, float) - np.asarray(axis, float) * L
    depth = float(np.dot(np.asarray(p_tcp, float) - p_pre, axis))
    return float(np.clip(depth / L, 0.0, 1.0) * T_dur)
