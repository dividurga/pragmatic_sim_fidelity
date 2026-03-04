"""Fridge reach task with Franka geometry and collision proxies, in 3D."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pinocchio as pin  # type: ignore
import pybullet_data  # type: ignore[import-untyped]

from ..core.types import State, Trajectory
from ..registry import register_task
from ..simulators.geom3d_sim.collision import collided_obstacles_only

AABB = Tuple[float, float, float, float, float, float]  # xmin,ymin,zmin,xmax,ymax,zmax


# ----------------------------
# Pinocchio Franka wrapper
# ----------------------------
class FrankaGeom3D:
    """Pinocchio-backed Franka kinematics with link-sphere collision proxies.

    Uses Panda URDF from pybullet_data.
    """

    def __init__(
        self,
        urdf_path: str,
        ee_frame: str = "panda_hand",
        base_translation: Tuple[float, float, float] = (-0.1, 0.95, -0.1),
        proxy_radius: float = 0.03,
        spheres_per_segment: int = 4,
    ):
        """Initialize the Franka geometry wrapper."""

        self.pin = pin
        self.model = pin.buildModelFromUrdf(urdf_path)  # pylint: disable=no-member
        self.data = self.model.createData()

        self.base_t = np.array(base_translation, dtype=float)
        self.proxy_r = float(proxy_radius)
        self.K = int(spheres_per_segment)

        # Identify the 7 Panda arm joints by name (robust)
        self.arm_joint_names = [f"panda_joint{i}" for i in range(1, 8)]
        self.arm_jids = []
        for name in self.arm_joint_names:
            jid = self.model.getJointId(name)
            if jid == 0:
                raise ValueError(
                    f"Joint '{name}' not found in URDF. Check URDF joint names."
                )
            self.arm_jids.append(jid)

        # Assume first 7 q entries correspond to panda_joint1..7
        self.nq_arm = 7
        if self.model.nq < self.nq_arm:
            raise ValueError(f"URDF model has nq={self.model.nq} < 7; wrong URDF?")

        self.lower = self.model.lowerPositionLimit[: self.nq_arm].copy()
        self.upper = self.model.upperPositionLimit[: self.nq_arm].copy()

        # EE frame id
        self.ee_fid = self.model.getFrameId(ee_frame)
        if self.ee_fid == len(self.model.frames):
            raise ValueError(f"EE frame '{ee_frame}' not found in URDF frames.")
        self.ee_frame = ee_frame

    def clamp_q(self, q: np.ndarray) -> np.ndarray:
        """Clamp a joint configuration to the robot's joint limits."""
        q = np.asarray(q, dtype=float).copy()
        return np.minimum(np.maximum(q, self.lower), self.upper)

    def _q_full(self, q_arm: np.ndarray) -> np.ndarray:
        """Convert a 7D arm joint configuration to the full robot configuration by
        padding with zeros for non-arm joints.

        we dont need this for the current task, but it can be useful for future
        extensions if we want to
        """
        q_full = np.zeros(self.model.nq, dtype=float)
        q_full[: self.nq_arm] = q_arm
        return q_full

    def fk_update(self, q_arm: np.ndarray) -> None:
        """Forward kinematics update for a given arm joint configuration."""
        q_full = self._q_full(q_arm)
        self.pin.forwardKinematics(  # pylint: disable=no-member
            self.model, self.data, q_full
        )
        self.pin.updateFramePlacements(  # pylint: disable=no-member
            self.model, self.data
        )

    def ee_pos(self, q_arm: np.ndarray) -> np.ndarray:
        """Get the world position for the end-effector frame given a 7D arm joint
        configuration."""
        self.fk_update(q_arm)
        oMf = self.data.oMf[self.ee_fid]
        return np.array(oMf.translation, dtype=float) + self.base_t

    def _joint_positions_plus_ee(self) -> List[np.ndarray]:
        """World positions of joint origins for panda_joint1..7 and then EE frame as
        last.

        Assumes fk_update already called.
        """
        pts: List[np.ndarray] = []
        for jid in self.arm_jids:
            oMi = self.data.oMi[jid]
            pts.append(np.array(oMi.translation, dtype=float) + self.base_t)
        pts.append(
            np.array(self.data.oMf[self.ee_fid].translation, dtype=float) + self.base_t
        )
        return pts

    def proxy_spheres(
        self, q_arm: np.ndarray, ee_radius: float
    ) -> List[Tuple[np.ndarray, float]]:
        """Build spheres along segments between consecutive joints and to EE.

        Also include an EE sphere with the provided ee_radius.
        """
        self.fk_update(q_arm)
        pts = self._joint_positions_plus_ee()

        spheres: List[Tuple[np.ndarray, float]] = []
        for a, b in zip(pts[:-1], pts[1:]):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            for k in range(1, self.K + 1):
                t = k / (self.K + 1)
                c = (1 - t) * a + t * b
                spheres.append((c, self.proxy_r))

        spheres.append((pts[-1], float(ee_radius)))
        return spheres

    def ik_position(
        self,
        target_world: np.ndarray,
        q_init: np.ndarray,
        iters: int = 120,
        tol: float = 1e-3,
        damping: float = 1e-2,
        step: float = 0.7,
    ) -> np.ndarray:
        """Damped least-squares IK for EE position only.

        target_world is in the SAME world frame as bounds/obstacles.
        """
        q = self.clamp_q(np.asarray(q_init, dtype=float).copy())

        # Convert world target -> model frame (undo base translation)
        target_model = np.asarray(target_world, dtype=float) - self.base_t

        for _ in range(iters):
            q_full = self._q_full(q)
            self.pin.forwardKinematics(  # pylint: disable=no-member
                self.model, self.data, q_full
            )
            self.pin.updateFramePlacements(  # pylint: disable=no-member
                self.model, self.data
            )

            oMf = self.data.oMf[self.ee_fid]
            cur = np.array(oMf.translation, dtype=float)
            err = target_model - cur

            if float(np.linalg.norm(err)) < tol:
                break

            J6 = self.pin.computeFrameJacobian(  # pylint: disable=no-member
                self.model,
                self.data,
                q_full,
                self.ee_fid,
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,  # pylint: disable=no-member
            )
            J = J6[:3, : self.nq_arm]  # 3x7

            A = J @ J.T + (damping**2) * np.eye(3)
            dq = J.T @ np.linalg.solve(A, err)

            q = self.clamp_q(q + step * dq)

        return q


# ----------------------------
# Task
# ----------------------------
@dataclass
class FridgeReachConfig:
    """Configuration for the fridge reach task."""

    ee_radius: float = 0.035
    goal_tol: float = 0.05
    step_size: float = 0.05
    w_collide: float = 1000.0
    w_step: float = 0.01

    base_translation: Tuple[float, float, float] = (-0.1, 0.95, -0.1)
    proxy_radius: float = 0.03
    spheres_per_segment: int = 4
    ee_frame: str = "panda_hand"


class FridgeReachFrankaGeom3D:
    """Fridge reach task with Franka geometry and collision proxies, in 3D."""

    name = "fridge_reach_franka_geom3d"

    def __init__(self, cfg: FridgeReachConfig | None = None):
        """Initialize the task with a given config."""
        self.cfg = cfg or FridgeReachConfig()
        self._robot: Optional[FrankaGeom3D] = None

    def _ensure_robot(self) -> None:
        """Initialize the Franka geometry wrapper if not already initialized."""
        if self._robot is not None:
            return

        urdf_path = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(
                f"Could not find Panda URDF at '{urdf_path}'. "
                "Your pybullet_data may not include Franka assets."
            )

        self._robot = FrankaGeom3D(
            urdf_path=urdf_path,
            ee_frame=self.cfg.ee_frame,
            base_translation=self.cfg.base_translation,
            proxy_radius=self.cfg.proxy_radius,
            spheres_per_segment=self.cfg.spheres_per_segment,
        )

    def reset(self, rng) -> State:
        """Reset the task to an initial state and return the initial state."""
        self._ensure_robot()
        assert self._robot is not None
        # im passing rng here for potential future
        # use in sampling randomized start states
        rng = rng  # pylint: disable=self-assigning-variable
        bounds: AABB = (0.0, 0.0, 0.0, 0.8, 0.6, 0.8)
        obstacles: List[AABB] = [
            (0.0, 0.0, 0.35, 0.80, 0.6, 0.38),
            (0.35, 0.22, 0.38, 0.43, 0.30, 0.65),
            (0.62, 0.45, 0.38, 0.70, 0.50, 0.50),
            (0.22, 0.33, 0.38, 0.26, 0.37, 0.50),
        ]

        goal_pos = np.array([0.55, 0.25, 0.55], dtype=float)

        # choose a start EE pose (outside the open front, higher to avoid shelf) ---
        start_ee = np.array([0.10, 0.70, 0.72], dtype=float)

        q_seed = np.array([0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.8], dtype=float)
        q0 = self._robot.ik_position(start_ee, q_seed)
        q0 = self._robot.clamp_q(q0)

        ee_pos = self._robot.ee_pos(q0)
        spheres = self._robot.proxy_spheres(q0, ee_radius=self.cfg.ee_radius)
        collided = collided_obstacles_only(spheres, obstacles)
        return {
            "q": q0,
            "ee_pos": ee_pos,
            "goal_pos": goal_pos,
            "ee_radius": float(self.cfg.ee_radius),
            "obstacles": obstacles,
            "bounds": bounds,
            "collided": bool(collided),
            "step_size": float(self.cfg.step_size),
            "w_step": float(self.cfg.w_step),
        }

    def step(self, state: State, action: np.ndarray) -> Tuple[State, Dict[str, Any]]:
        """Apply a delta_q action to the current state and return the next state and
        info."""
        self._ensure_robot()
        assert self._robot is not None

        q = np.asarray(state["q"], dtype=np.float64)
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (7,):
            raise ValueError(f"Expected action shape (7,), got {a.shape}")

        max_norm = 1.0  # tune
        n = float(np.linalg.norm(a))
        if n > max_norm:
            a *= max_norm / n

        q_next = self._robot.clamp_q(q + float(state["step_size"]) * a)

        ee_pos = self._robot.ee_pos(q_next)
        spheres = self._robot.proxy_spheres(q_next, ee_radius=float(state["ee_radius"]))
        collided = collided_obstacles_only(spheres, state["obstacles"])

        next_state = dict(state)
        next_state["q"] = q_next
        next_state["ee_pos"] = ee_pos
        next_state["collided"] = bool(collided)

        info: Dict[str, Any] = {
            "collided": bool(collided),
            "step_cost": float(state.get("w_step", 0.0)),
            "ee_pos": ee_pos,
        }
        return next_state, info

    def is_success(self, state: State) -> bool:
        """Check if the current state is a success (EE within goal_tol of goal and not
        collided)."""
        if state.get("collided", False):
            return False
        d = float(
            np.linalg.norm(np.asarray(state["ee_pos"]) - np.asarray(state["goal_pos"]))
        )
        return d < self.cfg.goal_tol

    def score(self, traj: Trajectory) -> float:
        """Compute a score for a trajectory based on final distance to goal, collisions,
        and step costs."""
        final = traj.states[-1]
        dist = float(
            np.linalg.norm(np.asarray(final["ee_pos"]) - np.asarray(final["goal_pos"]))
        )
        collided = any(info.get("collided", False) for info in traj.infos)
        step_cost = sum(float(info.get("step_cost", 0.0)) for info in traj.infos)
        return dist + (self.cfg.w_collide if collided else 0.0) + step_cost


@register_task("fridge_reach_franka_geom3d")
def _make_task():
    return FridgeReachFrankaGeom3D()
