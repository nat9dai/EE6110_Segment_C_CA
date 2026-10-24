"""Measure the Panda hand and finger collision boxes (TCP frame) into safegrasp/kinematics/panda_gripper_geometry.json.

    ~/isaacsim/python.sh scripts/measure_gripper_geometry.py
"""
import json
import os
import sys

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from safegrasp.kinematics.panda import FINGER_Z, FLANGE_Z, MAX_WIDTH, TCP_OFFSET, rotz  # noqa: E402

SPHERES_JSON = os.path.join(ROOT, "safegrasp", "kinematics", "panda_spheres.json")
OUT_JSON = os.path.join(ROOT, "safegrasp", "kinematics", "panda_gripper_geometry.json")
PAD_TOL = 5e-4       # inner-face points within 0.5 mm of the pad plane count as pad


def sym(lo, hi):
    """Symmetrise an interval about 0, so swapping the fingers changes nothing."""
    m = np.maximum(np.abs(lo), np.abs(hi))
    return -m, m


def main():
    with open(SPHERES_JSON) as f:
        hulls = {k: np.asarray(v, float) for k, v in json.load(f)["hulls"].items()}
    r6 = lambda a: np.round(np.asarray(a, float), 5).tolist()          # noqa: E731

    h = hulls["panda_hand"] - [0.0, 0.0, TCP_OFFSET]
    lo, hi = h.min(0), h.max(0)
    (plx, phx), (ply, phy) = sym(lo[0], hi[0]), sym(lo[1], hi[1])
    palm_lo, palm_hi = [plx, ply, lo[2]], [phx, phy, hi[2]]

    # wrist (link7): p_hand = Rz(pi/4) (p_7 - (0, 0, 0.107))
    w = (hulls["panda_link7"] - [0.0, 0.0, FLANGE_Z]) @ rotz(np.pi / 4).T - [0.0, 0.0, TCP_OFFSET]
    wlo, whi = w.min(0), w.max(0)
    (wlx, whx), (wly, why) = sym(wlo[0], whi[0]), sym(wlo[1], whi[1])
    wrist_lo, wrist_hi = [wlx, wly, wlo[2]], [whx, why, whi[2]]

    # +y finger = panda_leftfinger; its y = 0 is the pad plane at hand y = +w/2
    f = hulls["panda_leftfinger"]
    flo, fhi = f.min(0), f.max(0)
    fx = sym(flo[0], fhi[0])
    fz = np.array([flo[2], fhi[2]]) + FINGER_Z - TCP_OFFSET
    finger_offset = float(flo[1])                       # pad face at y = w/2 + offset
    finger_thickness = float(fhi[1] - flo[1])

    # pad z-range: hull slices whose inner face reaches the pad plane
    mesh = trimesh.convex.convex_hull(f)
    zs = np.linspace(flo[2] + 1e-5, fhi[2] - 1e-5, 1200)
    on_pad = []
    for z in zs:
        s = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        on_pad.append(s is not None and s.vertices[:, 1].min() <= flo[1] + PAD_TOL)
    zp = zs[np.asarray(on_pad)]
    pad_z = np.array([zp.min(), zp.max()]) + FINGER_Z - TCP_OFFSET

    out = {
        "description": "Franka hand collision geometry in the TCP frame (TCP = panda_hand + 0.1034 m along "
                       "hand z; x = hand x, y = closing axis, z = approach). AABBs of the Isaac Lab "
                       "FRANKA_PANDA_HIGH_PD_CFG collision hulls, symmetrised in x/y. Metres.",
        "tcp_offset": TCP_OFFSET,
        "max_width": MAX_WIDTH,
        "palm_lo": r6(palm_lo), "palm_hi": r6(palm_hi),
        "wrist_lo": r6(wrist_lo), "wrist_hi": r6(wrist_hi),
        # -y finger is the mirror image
        "finger_x": r6(fx), "finger_z": r6(fz),
        "finger_offset": round(finger_offset, 5), "finger_thickness": round(finger_thickness, 5),
        "pad_z": r6(pad_z),
        "notes": {
            "width": "w = pad-to-pad distance = 2 x finger joint position (finger_joint1/2 in [0, 0.04]); "
                     "verified in Isaac: finger body origin at hand y = +-q_f, pad plane at finger-frame y = 0",
            "tcp_vs_pads": f"TCP (z = 0, midway between the pad planes) lies on the flat pad face, which spans "
                           f"TCP z [{pad_z[0]:.4f}, {pad_z[1]:.4f}] (centre {pad_z.mean():.4f}); fingertips at "
                           f"TCP z = {fz[1]:.4f} (TCP is {1e3 * fz[1]:.1f} mm behind the fingertips), finger "
                           f"root at z = {fz[0]:.4f}; the finger boxes overlap the palm by "
                           f"{1e3 * (palm_hi[2] - fz[0]):.1f} mm in z",
            "pad_tolerance_m": PAD_TOL,
            "palm_raw_y": r6([lo[1], hi[1]]),
            "wrist_raw_xy": r6([[wlo[0], whi[0]], [wlo[1], whi[1]]]),
            "finger_raw_x": r6([flo[0], fhi[0]]),
        },
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
