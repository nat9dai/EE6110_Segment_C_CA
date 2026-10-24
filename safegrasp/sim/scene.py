"""Isaac Lab scene: Franka Panda, table (ground plane, z = 0), RGB-D cameras and a pool of GSO objects.

Unused objects are parked out of view, so scenes change by teleporting. Import after ``AppLauncher`` starts Kit."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.sensors import Camera, CameraCfg, ContactSensor, ContactSensorCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from safegrasp.perception.types import View

ROBOT_BODIES = ["panda_link1", "panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6",
                "panda_link7", "panda_hand", "panda_leftfinger", "panda_rightfinger"]
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]

# name -> (eye, look-at), world frame [m]
CAMERAS = {
    "main": ((1.20, 0.0, 0.33), (0.50, 0.0, 0.06)),   # ~21 deg elevation, long shadows
    "left": ((0.55, 0.60, 0.55), (0.55, 0.0, 0.05)),
    "right": ((0.55, -0.60, 0.55), (0.55, 0.0, 0.05)),
    "close": ((0.55, 0.35, 0.20), (0.55, 0.0, 0.05)),   # close-up camera, figures only
}


@dataclass
class SceneCfg:
    pool: list = field(default_factory=list)      # list[GSOEntry]
    device: str = "cpu"
    dt: float = 1.0 / 120.0
    decimation: int = 2                           # control at 60 Hz
    cam_w: int = 640
    cam_h: int = 480
    cameras: tuple = ("main", "left", "right")
    contact_sensing: bool = True
    park_origin: tuple = (-30.0, 30.0)
    obj_pos_iters: int = 16        # PhysX iterations, as in the Isaac Lab lift task
    obj_vel_iters: int = 1


class GraspScene:
    def __init__(self, cfg: SceneCfg):
        self.cfg = cfg
        self.K = len(cfg.pool)
        sim_cfg = sim_utils.SimulationCfg(dt=cfg.dt, device=cfg.device, render_interval=cfg.decimation)
        sim_cfg.physx.bounce_threshold_velocity = 0.01
        sim_cfg.physx.friction_correlation_distance = 0.00625
        self.sim = sim_utils.SimulationContext(sim_cfg)
        self.sim.set_camera_view((1.6, 0.9, 1.0), (0.5, 0.0, 0.1))

        sim_utils.GroundPlaneCfg(color=(0.33, 0.36, 0.40),
                                 physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7)
                                 ).func("/World/ground", sim_utils.GroundPlaneCfg(color=(0.33, 0.36, 0.40)))
        sim_utils.DomeLightCfg(intensity=1500.0, color=(0.95, 0.95, 1.0)).func("/World/dome", sim_utils.DomeLightCfg(intensity=1500.0))
        sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0).func("/World/sun", sim_utils.DistantLightCfg(intensity=2500.0))

        rcfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="/World/Robot")
        rcfg.spawn.activate_contact_sensors = cfg.contact_sensing
        self.robot = Articulation(rcfg)

        objs = {}
        for i, e in enumerate(cfg.pool):
            objs[f"obj_{i:03d}"] = RigidObjectCfg(
                prim_path=f"/World/Objects/obj_{i:03d}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=e.usd,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=cfg.obj_pos_iters, solver_velocity_iteration_count=cfg.obj_vel_iters,
                        max_depenetration_velocity=1.0)),
                init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(self._park_pos(i)), rot=(1.0, 0.0, 0.0, 0.0)),
            )
        self.objects = RigidObjectCollection(RigidObjectCollectionCfg(rigid_objects=objs))

        self.cams: dict[str, Camera] = {}
        for name in cfg.cameras:
            self.cams[name] = Camera(CameraCfg(
                prim_path=f"/World/Cam_{name}", height=cfg.cam_h, width=cfg.cam_w,
                data_types=["distance_to_image_plane", "rgb", "instance_id_segmentation_fast"],
                colorize_instance_id_segmentation=False, update_period=0.0,
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.05, 3.0)),
            ))

        self.contacts: dict[str, ContactSensor] = {}
        if cfg.contact_sensing and self.K > 0:
            for b in ROBOT_BODIES:
                self.contacts[b] = ContactSensor(ContactSensorCfg(
                    prim_path=f"/World/Robot/{b}", filter_prim_paths_expr=[f"/World/Objects/obj_{i:03d}" for i in range(self.K)],
                    update_period=0.0, history_length=0))

        self.sim.reset()
        self.device = self.sim.device
        for name, (eye, tgt) in CAMERAS.items():
            if name in self.cams:
                self.cams[name].set_world_poses_from_view(torch.tensor([eye], device=self.device, dtype=torch.float32),
                                                          torch.tensor([tgt], device=self.device, dtype=torch.float32))
                self.cams[name].reset()  # set_world_poses_from_view does not refresh data.pos_w / quat_w_ros
        self.arm_ids = [self.robot.joint_names.index(j) for j in ARM_JOINTS]
        self.finger_ids = [self.robot.joint_names.index(j) for j in ("panda_finger_joint1", "panda_finger_joint2")]
        self.body_ids = {b: self.robot.body_names.index(b) for b in self.robot.body_names}
        self.pool_extents = np.array([e.extents for e in cfg.pool], dtype=float).reshape(-1, 3)
        self.pool_points = [e.load_points().astype(np.float64) for e in cfg.pool]
        self.active = np.zeros(self.K, dtype=bool)
        self.sim_time = 0.0
        self._q_target = self.robot.data.default_joint_pos.clone()

    def _park_pos(self, i):
        x0, y0 = self.cfg.park_origin
        return np.array([x0 - 0.6 * (i % 10), y0 + 0.6 * (i // 10), 0.3])

    def _write_objects(self, poses: np.ndarray):
        t = torch.tensor(poses[None], device=self.device, dtype=torch.float32)
        self.objects.write_object_pose_to_sim(t)
        self.objects.write_object_velocity_to_sim(torch.zeros(1, self.K, 6, device=self.device))

    def object_poses(self) -> np.ndarray:
        """(K,7) GT poses [pos, quat(w,x,y,z)] of all pool objects."""
        d = self.objects.data
        return torch.cat([d.object_pos_w[0], d.object_quat_w[0]], dim=-1).cpu().numpy().astype(np.float64)

    def object_speeds(self) -> np.ndarray:
        d = self.objects.data
        return torch.linalg.norm(d.object_lin_vel_w[0], dim=-1).cpu().numpy()

    def place_scene(self, spec, settle_time=2.5, settle_speed=0.01):
        """Teleport ``spec``'s objects into the workspace, let them settle and return the (K,7) pose snapshot."""
        poses = np.stack([np.r_[self._park_pos(i), 1, 0, 0, 0] for i in range(self.K)])
        self.active[:] = False
        for p in spec.placements:
            poses[p.slot, :3] = p.pos
            poses[p.slot, 3:] = p.quat
            self.active[p.slot] = True
        # swing the arm aside so objects from tall piles cannot come to rest on the hand
        q_aside = self.robot_home_q().copy()
        q_aside[0] += 1.6
        self.robot_reset(q_aside, 0.08)
        self._write_objects(poses)
        t = 0.0
        while t < settle_time:
            self.step_physics(self.cfg.decimation)
            t += self.cfg.decimation * self.cfg.dt
            if t > 0.6 and self.object_speeds()[self.active].max(initial=0.0) < settle_speed:
                break
        self.robot_reset(self.robot_home_q(), 0.08)
        snap = self.object_poses()
        # repeat restore -> simulate until at rest: paired trials need an equilibrium start
        for _ in range(3):
            self.restore(snap)
            self.step_physics(int(1.0 / self.cfg.dt))
            new = self.object_poses()
            drift = np.linalg.norm(new[self.active, :3] - snap[self.active, :3], axis=1).max(initial=0.0)
            snap = new
            if drift < 0.003:
                break
        return snap

    def null_drift(self, snapshot, duration=8.0) -> np.ndarray:
        """Return (K,) displacement [m] of each object after ``duration`` s with the robot idle at home."""
        self.restore(snapshot)
        self.step_physics(int(duration / self.cfg.dt))
        return np.linalg.norm(self.object_poses()[:, :3] - snapshot[:, :3], axis=1)

    def restore(self, snapshot: np.ndarray, q=None, width=0.08, settle_steps=6):
        """Restore an exact object configuration and the robot home pose (for paired trials)."""
        self.robot_reset(self.robot_home_q() if q is None else q, width)
        self._write_objects(snapshot)
        self.step_physics(settle_steps)

    def robot_home_q(self):
        from safegrasp.kinematics.panda import Panda  # noqa: lazy (pure numpy)
        if not hasattr(self, "_home"):
            self._home = Panda().q_home.copy()
        return self._home

    def robot_reset(self, q_arm, width=0.08):
        q = self.robot.data.default_joint_pos.clone()
        q[0, self.arm_ids] = torch.tensor(q_arm, device=self.device, dtype=q.dtype)
        q[0, self.finger_ids] = 0.5 * float(width)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self._q_target = q.clone()
        self.robot.set_joint_position_target(q)
        self.robot.set_joint_velocity_target(torch.zeros_like(q))
        self.robot.write_data_to_sim()
        self.robot.update(0.0)

    def robot_q(self) -> np.ndarray:
        return self.robot.data.joint_pos[0, self.arm_ids].cpu().numpy().astype(np.float64)

    def robot_qd(self) -> np.ndarray:
        return self.robot.data.joint_vel[0, self.arm_ids].cpu().numpy().astype(np.float64)

    def gripper_width(self) -> float:
        return float(self.robot.data.joint_pos[0, self.finger_ids].sum())

    def body_pose(self, name) -> np.ndarray:
        i = self.body_ids[name]
        return torch.cat([self.robot.data.body_pos_w[0, i], self.robot.data.body_quat_w[0, i]]).cpu().numpy()

    def set_targets(self, q_arm, width, qd_arm=None):
        self._q_target[0, self.arm_ids] = torch.tensor(q_arm, device=self.device, dtype=self._q_target.dtype)
        self._q_target[0, self.finger_ids] = 0.5 * float(width)
        self.robot.set_joint_position_target(self._q_target)
        vt = torch.zeros_like(self._q_target)
        if qd_arm is not None:  # feed-forward velocity target (PD: kp*(q*-q) + kd*(qd*-qd))
            vt[0, self.arm_ids] = torch.tensor(qd_arm, device=self.device, dtype=vt.dtype)
        self.robot.set_joint_velocity_target(vt)

    def step_physics(self, n=1):
        for _ in range(n):
            self.robot.write_data_to_sim()
            self.sim.step(render=False)
            self.robot.update(self.cfg.dt)
            self.objects.update(self.cfg.dt)
            for c in self.contacts.values():
                c.update(self.cfg.dt)
            self.sim_time += self.cfg.dt

    def control_step(self, q_arm_target, width, qd_arm=None):
        self.set_targets(q_arm_target, width, qd_arm)
        self.step_physics(self.cfg.decimation)

    def contact_matrix(self) -> np.ndarray:
        """Return (n_robot_bodies, K) contact force magnitudes [N] between robot bodies and pool objects."""
        if not self.contacts:
            return np.zeros((len(ROBOT_BODIES), self.K))
        out = []
        for b in ROBOT_BODIES:
            fm = self.contacts[b].data.force_matrix_w  # (1, 1, M, 3)
            out.append(torch.linalg.norm(fm[0, 0], dim=-1).cpu().numpy())
        return np.stack(out)

    def look_at(self, name, eye, target):
        """Point camera ``name`` at ``target`` from ``eye``."""
        cam = self.cams[name]
        cam.set_world_poses_from_view(torch.tensor([eye], device=self.device, dtype=torch.float32),
                                      torch.tensor([target], device=self.device, dtype=torch.float32))
        cam.reset()

    def capture(self, names=None, n_render=3, with_rgb=False):
        """Render and return a list of :class:`View` (+ RGB images if requested)."""
        names = list(self.cams) if names is None else list(names)
        for _ in range(n_render):
            self.sim.render()
        views, rgbs = [], []
        for name in names:
            cam = self.cams[name]
            cam.update(0.0, force_recompute=True)
            depth = cam.data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy().astype(np.float32)
            depth[~np.isfinite(depth)] = 0.0
            seg_raw = cam.data.output["instance_id_segmentation_fast"][0, ..., 0].cpu().numpy()
            info = cam.data.info[0]["instance_id_segmentation_fast"]["idToLabels"]
            lut = {}
            for k, path in info.items():
                kid = int(k)
                if "/World/Objects/obj_" in path:
                    lut[kid] = int(path.split("/World/Objects/obj_")[1][:3])
                elif path.startswith("/World/Robot"):
                    lut[kid] = -2
                else:
                    lut[kid] = -1
            seg = np.full(seg_raw.shape, -1, dtype=np.int32)
            for kid, v in lut.items():
                if v != -1:
                    seg[seg_raw == kid] = v
            K = cam.data.intrinsic_matrices[0].cpu().numpy().astype(np.float64)
            pos = cam.data.pos_w[0].cpu().numpy().astype(np.float64)
            q = cam.data.quat_w_ros[0].cpu().numpy().astype(np.float64)
            T = np.eye(4)
            T[:3, :3] = _quat_to_R(q)
            T[:3, 3] = pos
            views.append(View(depth=depth, K=K, T_wc=T, seg=seg, name=name))
            if with_rgb:
                rgbs.append(cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
        return (views, rgbs) if with_rgb else views

    def object_points_world(self, slots, poses=None) -> list[np.ndarray]:
        poses = self.object_poses() if poses is None else poses
        out = []
        for s in slots:
            R = _quat_to_R(poses[s, 3:])
            out.append(self.pool_points[s] @ R.T + poses[s, :3])
        return out


def _quat_to_R(q) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
