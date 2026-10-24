"""Shared Times New Roman font (or a same-width fallback) for every figure."""
from __future__ import annotations

import os

SERIF = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
_TTF = [
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
]


def apply():
    """Set the Times fonts; call after any other rcParams update."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "stix"})


def ttf_path() -> str | None:
    """Path of a Times(-compatible) TrueType file for PIL text, or None."""
    return next((p for p in _TTF if os.path.exists(p)), None)
