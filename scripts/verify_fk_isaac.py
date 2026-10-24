"""Check the numpy Panda kinematics (poses, Jacobians, limits) against Isaac Lab's Franka.

    timeout 900 ~/isaacsim/python.sh -u scripts/verify_fk_isaac.py [--n 200] [--dump-collision out.npz] [--save FILE.npz]
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200, help="number of random configurations")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dump-collision", default=None, help="write collision meshes (link frame) to this npz")
ap.add_argument("--save", default=None, help="save the Isaac ground truth (q, body poses, COMs, Jacobians) "
                                             "to this npz, e.g. tests/fixtures/isaac_franka_fk.npz")
args = ap.parse_args()

t0 = time.time()
from isaaclab.app import AppLauncher  # noqa: E402

app = AppLauncher(headless=True).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # noqa: E402

from safegrasp.kinematics.panda import (Panda, LINK_NAMES, TCP_OFFSET, quat_to_matrix,  # noqa: E402
                                        rot_log, matrix_to_quat)


def dump_collision(path, robot):
    """Save each body's collision mesh (vertices, faces) in its own frame."""
    from pxr import Usd, UsdGeom
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = {}
    for body in robot.body_names:
        link_prim = stage.GetPrimAtPath(f"/World/Robot/{body}")
        V, F, nv = [], [], 0
        for prim in Usd.PrimRange(link_prim, Usd.TraverseInstanceProxies()):
            if "collision" not in str(prim.GetPath()).lower() or not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            pts = np.array(mesh.GetPointsAttr().Get(), dtype=np.float64)
            cnt = np.array(mesh.GetFaceVertexCountsAttr().Get())
            idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
            M, _ = xc.ComputeRelativeTransform(prim, link_prim)
            T = np.array(M, dtype=np.float64).T               # Gf matrices are row-vector convention
            tris, off = [], 0
            for c in cnt:                                     # fan triangulation
                tris += [[idx[off], idx[off + k], idx[off + k + 1]] for k in range(1, c - 1)]
                off += c
            V.append(pts @ T[:3, :3].T + T[:3, 3])
            F.append(np.array(tris) + nv)
            nv += len(pts)
        if V:
            out[f"{body}__v"], out[f"{body}__f"] = np.concatenate(V), np.concatenate(F)
    np.savez(path, **out)
    print(f"collision meshes of {len(out) // 2} bodies written to {path}", flush=True)


def main():
    dt = 1.0 / 120.0
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cpu"))
    robot = Articulation(FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    print(f"setup {time.time() - t0:.1f} s", flush=True)
    if args.dump_collision:
        dump_collision(args.dump_collision, robot)

    pd = Panda()
    names = list(robot.body_names)
    assert set(names) == set(LINK_NAMES), names
    lim = robot.data.joint_pos_limits[0].numpy().astype(np.float64)
    vlim = robot.data.joint_vel_limits[0].numpy().astype(np.float64)
    print("root pos", robot.data.root_pos_w[0].numpy(), "quat", robot.data.root_quat_w[0].numpy())
    print(f"joint limit |err| max: lower {np.abs(lim[:7, 0] - pd.q_lower).max():.2e}  "
          f"upper {np.abs(lim[:7, 1] - pd.q_upper).max():.2e} rad;  finger limits {lim[7:].tolist()}")
    print(f"vel limit Isaac {np.round(vlim[:7], 4).tolist()}  ->  Panda.qd_max = 0.8 x "
          f"{np.round(pd.qd_max / 0.8, 4).tolist()}  (|err| {np.abs(vlim[:7] - pd.qd_max / 0.8).max():.2e})")

    rng = np.random.default_rng(args.seed)
    lo, hi = lim[:, 0], lim[:, 1]
    Q = lo + (hi - lo) * rng.uniform(0.02, 0.98, size=(args.n, 9))
    Q[0, :7], Q[0, 7:] = pd.q_home, 0.04
    body_ids = [names.index(n) for n in LINK_NAMES]
    ep = np.zeros((args.n, len(LINK_NAMES)))
    er = np.zeros_like(ep)
    ejl = np.zeros_like(ep)
    eja = np.zeros_like(ep)
    gt = {k: [] for k in ("q", "pos", "quat", "com", "jac")}
    for n in range(args.n):
        qt = torch.tensor(Q[n], dtype=torch.float32)[None]
        robot.write_joint_state_to_sim(qt, torch.zeros_like(qt))
        robot.set_joint_position_target(qt)
        robot.write_data_to_sim()
        sim.step(render=False)
        robot.update(dt)
        q = robot.data.joint_pos[0].numpy().astype(np.float64)   # compare at the state actually read
        P = robot.data.body_pos_w[0].numpy().astype(np.float64)
        R = quat_to_matrix(robot.data.body_quat_w[0].numpy().astype(np.float64))
        Jis = robot.root_physx_view.get_jacobians()[0].numpy().astype(np.float64)  # (nb-1, 6, 9)
        Pcom = robot.data.body_com_pos_w[0].numpy().astype(np.float64)  # PhysX linear J is at the COM
        for k, v in zip(gt, (q, P, robot.data.body_quat_w[0].numpy(), Pcom, Jis)):
            gt[k].append(np.asarray(v, dtype=np.float32))
        # each finger joint moves its own finger, so width = 2 q_f
        T_l = pd.link_transforms(q[:7], width=2 * q[7])
        T_r = pd.link_transforms(q[:7], width=2 * q[8])
        z = T_l[1:8, :3, 2]
        for k, (bn, bi) in enumerate(zip(LINK_NAMES, body_ids)):
            Tn = T_r[k] if bn == "panda_rightfinger" else T_l[k]
            ep[n, k] = np.linalg.norm(Tn[:3, 3] - P[bi])
            er[n, k] = np.linalg.norm(rot_log(Tn[:3, :3].T @ R[bi]))
            if bi == 0:
                continue                                     # fixed root has no Jacobian row
            Jl = pd.jacobian_point(q[:7], bn, Pcom[bi], width=2 * q[7])
            Ja = z.T * (np.arange(7) < pd._n_joints_of_link[k])
            ejl[n, k] = np.abs(Jl - Jis[bi - 1, :3, :7]).max()
            eja[n, k] = np.abs(Ja - Jis[bi - 1, 3:, :7]).max()
        if n == 0:
            Th = np.eye(4)
            Th[:3, :3], Th[:3, 3] = R[names.index("panda_hand")], P[names.index("panda_hand")]
            tcp = Th[:3, 3] + TCP_OFFSET * Th[:3, 2]
            print(f"q_home: Isaac TCP {np.round(tcp, 5).tolist()}  hand quat "
                  f"{np.round(matrix_to_quat(Th[:3, :3]), 4).tolist()}  numpy TCP "
                  f"{np.round(pd.tcp(q[:7])[:3, 3], 5).tolist()}")

    print(f"\n{args.n} configurations; max over configurations:")
    print(f"{'body':18s} {'pos [m]':>10s} {'rot [rad]':>10s} {'J_lin':>10s} {'J_ang':>10s}")
    for k, bn in enumerate(LINK_NAMES):
        print(f"{bn:18s} {ep[:, k].max():10.2e} {er[:, k].max():10.2e} {ejl[:, k].max():10.2e} {eja[:, k].max():10.2e}")
    print(f"MAX_POS_ERR {ep.max():.3e} m   MAX_ROT_ERR {er.max():.3e} rad   "
          f"MAX_JAC_LIN_ERR {ejl.max():.3e}   MAX_JAC_ANG_ERR {eja.max():.3e}")
    ok = ep.max() < 1e-4 and er.max() < 1e-3
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        np.savez_compressed(args.save, **{k: np.stack(v) for k, v in gt.items()}, body_names=np.array(names),
                            joint_names=np.array(robot.joint_names), limits=lim, vel_limits=vlim)
        print("ground truth saved to", args.save)
    print("FK_MATCH", "PASS" if ok else "FAIL", f"(tolerance 1e-4 m / 1e-3 rad)  total {time.time() - t0:.1f} s")


main()
sys.stdout.flush()
os._exit(0)
