"""Debug visualization for 3D geometric sim episodes."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

AABB = Tuple[float, float, float, float, float, float]


# ----------------------------
# Small helpers
# ----------------------------
def _project_aabb_xy(aabb: AABB):
    xmin, ymin, _, xmax, ymax, _ = aabb
    return xmin, ymin, xmax - xmin, ymax - ymin


def _project_aabb_yz(aabb: AABB):
    _, ymin, zmin, _, ymax, zmax = aabb
    return ymin, zmin, ymax - ymin, zmax - zmin


def _draw_scene_xy_yz(ax_xy, ax_yz, bounds: AABB, obstacles: Sequence[AABB]):
    bxmin, bymin, bzmin, bxmax, bymax, bzmax = bounds

    ax_xy.set_title("Top-down (x-y)")
    ax_xy.set_xlim(bxmin, bxmax)
    ax_xy.set_ylim(bymin, bymax)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.add_patch(Rectangle((bxmin, bymin), bxmax - bxmin, bymax - bymin, fill=False))
    for aabb in obstacles:
        x, y, w, h = _project_aabb_xy(aabb)
        ax_xy.add_patch(Rectangle((x, y), w, h, fill=False))

    ax_yz.set_title("Side view (y-z)")
    ax_yz.set_xlim(bymin, bymax)
    ax_yz.set_ylim(bzmin, bzmax)
    ax_yz.set_aspect("equal", adjustable="box")
    ax_yz.add_patch(Rectangle((bymin, bzmin), bymax - bymin, bzmax - bzmin, fill=False))
    for aabb in obstacles:
        y, z, w, h = _project_aabb_yz(aabb)
        ax_yz.add_patch(Rectangle((y, z), w, h, fill=False))


def _draw_arm_xy_yz(
    ax_xy,
    ax_yz,
    pts: np.ndarray,  # (N,3)
    joint_radius: float = 0.02,
    alpha: float = 0.12,
):
    """“PyBullet-ish” look: links as polylines + joints as circles."""
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"pts must be (N,3), got {pts.shape}")

    ax_xy.plot(pts[:, 0], pts[:, 1], linewidth=1, alpha=alpha)
    ax_yz.plot(pts[:, 1], pts[:, 2], linewidth=1, alpha=alpha)

    for pt in pts:
        ax_xy.add_patch(
            Circle((float(pt[0]), float(pt[1])), joint_radius, fill=False, alpha=alpha)
        )
        ax_yz.add_patch(
            Circle((float(pt[1]), float(pt[2])), joint_radius, fill=False, alpha=alpha)
        )


# ----------------------------
# Main plotter
# ----------------------------
def plot_episode_xy_yz(
    states: List[dict],
    title: str = "Episode",
    fk_points_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    arm_stride: int = 8,
    arm_alpha: float = 0.10,
    joint_radius: float = 0.02,
):
    """
    Plots:
      - XY (top-down) obstacles + end-effector trajectory
      - YZ (side)     obstacles + end-effector trajectory
    Optional:
      - Arm overlay via fk_points_fn(q)->(N,3) points.

    Required state keys:
      states[t]["ee_pos"]   : (3,)
      states[0]["goal_pos"] : (3,)
      states[0]["bounds"]   : AABB
      states[0]["obstacles"]: list[AABB]
    Optional for overlay:
      states[t]["q"]        : (7,)
    """
    if not states:
        raise ValueError("states is empty")

    ee = np.asarray([s["ee_pos"] for s in states], dtype=float)
    goal = np.asarray(states[0]["goal_pos"], dtype=float)
    bounds = tuple(states[0]["bounds"])
    obstacles = list(states[0]["obstacles"])

    fig = plt.figure(figsize=(10, 4))
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_yz = fig.add_subplot(1, 2, 2)

    _draw_scene_xy_yz(ax_xy, ax_yz, bounds, obstacles)

    ax_xy.plot(ee[:, 0], ee[:, 1], marker="o", linewidth=1)
    ax_xy.scatter([ee[0, 0]], [ee[0, 1]], marker="s", label="start")
    ax_xy.scatter([goal[0]], [goal[1]], marker="*", label="goal")

    ax_yz.plot(ee[:, 1], ee[:, 2], marker="o", linewidth=1)
    ax_yz.scatter([ee[0, 1]], [ee[0, 2]], marker="s", label="start")
    ax_yz.scatter([goal[1]], [goal[2]], marker="*", label="goal")

    if fk_points_fn is not None and arm_stride > 0 and "q" in states[0]:
        for t in range(0, len(states), arm_stride):
            q = np.asarray(states[t]["q"], dtype=float)
            pts = np.asarray(fk_points_fn(q), dtype=float)
            _draw_arm_xy_yz(
                ax_xy, ax_yz, pts, joint_radius=joint_radius, alpha=arm_alpha
            )
        ax_xy.plot([], [], linestyle="none", marker="o", alpha=arm_alpha, label="arm")

    ax_xy.legend(loc="best")
    ax_yz.legend(loc="best")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    return fig, (ax_xy, ax_yz)


# ----------------------------
# Franka FK provider (PyBullet) + Panda helpers
# ----------------------------
def infer_franka_panda_7dof_and_chain_to_hand(
    p,
    robot_id: int,
    hand_link_name: str = "panda_hand",
    physics_client_id: Optional[int] = None,
) -> Tuple[List[int], List[int]]:
    """
    Robustly infers:
      - the 7 revolute arm joint indices (panda_joint1..panda_joint7)
      - a link chain (joint child link indices) ending at `hand_link_name`

    If your URDF uses a typo like "panda_hand", pass hand_link_name="panda_hand".
    """
    n = (
        p.getNumJoints(robot_id)
        if physics_client_id is None
        else p.getNumJoints(robot_id, physicsClientId=physics_client_id)
    )

    name_to_link = {}
    parent_of = {}
    revolute = []

    for ji in range(n):
        info = (
            p.getJointInfo(robot_id, ji)
            if physics_client_id is None
            else p.getJointInfo(robot_id, ji, physicsClientId=physics_client_id)
        )
        jname = info[1].decode("utf-8")
        jtype = info[2]
        parent_link = info[16]
        child_link = ji  # PyBullet convention

        name_to_link[jname] = child_link
        parent_of[child_link] = parent_link

        if jtype == p.JOINT_REVOLUTE:
            revolute.append(ji)

    # Prefer canonical names if present; else fallback to first 7 revolute joints.
    canonical = []
    for k in range(1, 8):
        nm = f"panda_joint{k}"
        if nm in name_to_link:
            canonical.append(int(nm.rsplit("joint", maxsplit=1)[-1]) - 1)
    # The above is intentionally not used; we instead search by joint name directly:
    canonical = []
    for ji in range(n):
        info = (
            p.getJointInfo(robot_id, ji)
            if physics_client_id is None
            else p.getJointInfo(robot_id, ji, physicsClientId=physics_client_id)
        )
        jname = info[1].decode("utf-8")
        jtype = info[2]
        if jtype == p.JOINT_REVOLUTE and jname.startswith("panda_joint"):
            canonical.append(ji)

    joint_indices = canonical[:7] if len(canonical) >= 7 else revolute[:7]
    if len(joint_indices) != 7:
        raise RuntimeError(f"Could not infer 7 arm joints. Found: {len(joint_indices)}")

    if hand_link_name not in name_to_link:
        # fallback: try common alternatives
        for alt in ["panda_hand", "hand", "panda_link8", "panda_link7"]:
            if alt in name_to_link:
                hand_link_name = alt
                break
        if hand_link_name not in name_to_link:
            # last resort: draw to last arm joint child link
            target = joint_indices[-1]
        else:
            target = name_to_link[hand_link_name]
    else:
        target = name_to_link[hand_link_name]

    # Build chain from target back to base (-1)
    chain = []
    cur = target
    while cur != -1:
        chain.append(cur)
        cur = parent_of.get(cur, -1)
    chain.reverse()

    return joint_indices, chain


def make_pybullet_franka_fk_points_fn(
    p,  # pybullet module
    robot_id: int,
    joint_indices: Sequence[int],
    link_indices: Sequence[int],
    physics_client_id: Optional[int] = None,
):
    """fk_points_fn(q)->(N,3):

    - reset joint states for the 7-DoF
    - return [base_pos] + positions of each link in `link_indices`
    """
    joint_indices = list(joint_indices)
    link_indices = list(link_indices)

    def fk_points_fn(q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != len(joint_indices):
            raise ValueError(
                f"Expected q of size {len(joint_indices)}, got {q.shape[0]}"
            )

        for ji, qv in zip(joint_indices, q):
            if physics_client_id is None:
                p.resetJointState(robot_id, ji, float(qv))
            else:
                p.resetJointState(
                    robot_id, ji, float(qv), physicsClientId=physics_client_id
                )

        if physics_client_id is None:
            base_pos, _ = p.getBasePositionAndOrientation(robot_id)
        else:
            base_pos, _ = p.getBasePositionAndOrientation(
                robot_id, physicsClientId=physics_client_id
            )

        pts = [np.asarray(base_pos, dtype=float)]

        for li in link_indices:
            if physics_client_id is None:
                ls = p.getLinkState(robot_id, li, computeForwardKinematics=True)
            else:
                ls = p.getLinkState(
                    robot_id,
                    li,
                    computeForwardKinematics=True,
                    physicsClientId=physics_client_id,
                )
            pts.append(np.asarray(ls[4], dtype=float))  # worldLinkFramePosition

        return np.vstack(pts)

    return fk_points_fn


def make_panda7_fk_fn_to_hand(
    p,
    robot_id: int,
    hand_link_name: str = "panda_hand",
    physics_client_id: Optional[int] = None,
):
    """
    Convenience wrapper:
      - infers 7-DoF joints + link chain ending at `hand_link_name`
      - returns fk_points_fn(q) for plotting
    """
    joint_idxs, link_chain = infer_franka_panda_7dof_and_chain_to_hand(
        p,
        robot_id=robot_id,
        hand_link_name=hand_link_name,
        physics_client_id=physics_client_id,
    )
    fk_fn = make_pybullet_franka_fk_points_fn(
        p,
        robot_id=robot_id,
        joint_indices=joint_idxs,
        link_indices=link_chain,
        physics_client_id=physics_client_id,
    )
    return fk_fn, joint_idxs, link_chain
