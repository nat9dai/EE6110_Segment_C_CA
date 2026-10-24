"""Perception data types: a :class:`View` is one calibrated depth + segmentation image.
Depth is camera-frame z [m] (not ray length); integer (u, v) are pixel centres; T_wc uses the ROS
optical frame (z forward, x right, y down). Robot pixels (seg -2) count as invalid depth."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SEG_BACKGROUND = -1  #: table / background / anything that is not a scene object
SEG_ROBOT = -2       #: robot pixels (invalid for mapping)


def _squeeze_channel(a: np.ndarray) -> np.ndarray:
    """(H,W,1) -> (H,W); other shapes unchanged."""
    return a[..., 0] if a.ndim == 3 and a.shape[-1] == 1 else a


@dataclass
class View:
    depth: np.ndarray    # (H,W) float32 metres, invalid = 0 or non-finite
    K: np.ndarray        # (3,3)
    T_wc: np.ndarray     # (4,4) world_from_camera, ROS optical frame
    seg: np.ndarray      # (H,W) int32 object index, -1 = not an object, -2 = robot
    name: str = ""       # optional label, e.g. 'main'
    meta: dict = field(default_factory=dict)  # optional extras (rgb, timestamps, ...)

    def __post_init__(self):
        # Isaac Lab camera outputs are (H, W, 1)
        self.depth = _squeeze_channel(np.asarray(self.depth, dtype=np.float32))
        if self.depth.ndim != 2:
            raise ValueError(f"depth must be (H,W) (or (H,W,1)), got {self.depth.shape}")
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.T_wc = np.asarray(self.T_wc, dtype=np.float64).reshape(4, 4)
        if self.seg is None:  # no segmentation available: everything is "background"
            self.seg = np.full(self.depth.shape, SEG_BACKGROUND, dtype=np.int32)
        else:
            self.seg = _squeeze_channel(np.asarray(self.seg, dtype=np.int32))
        if self.seg.shape != self.depth.shape:
            raise ValueError(f"seg shape {self.seg.shape} != depth shape {self.depth.shape}")

    @property
    def height(self) -> int:
        return int(self.depth.shape[0])

    @property
    def width(self) -> int:
        return int(self.depth.shape[1])

    @property
    def cam_center(self) -> np.ndarray:
        """Camera optical centre in world coordinates, (3,)."""
        return self.T_wc[:3, 3].copy()

    @property
    def T_cw(self) -> np.ndarray:
        """camera_from_world (inverse of ``T_wc``), (4,4)."""
        R, t = self.T_wc[:3, :3], self.T_wc[:3, 3]
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t
        return T

    def valid_mask(self, include_robot: bool = False) -> np.ndarray:
        """(H,W) bool: finite, positive depth (and not a robot pixel unless ``include_robot``)."""
        m = np.isfinite(self.depth) & (self.depth > 0)
        if not include_robot:
            m &= self.seg != SEG_ROBOT
        return m
