from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt

AABB = Tuple[float, float, float, float, float, float]


def _project_aabb_xy(aabb: AABB):
    xmin, ymin, zmin, xmax, ymax, zmax = aabb
    return xmin, ymin, xmax - xmin, ymax - ymin


def _project_aabb_yz(aabb: AABB):
    xmin, ymin, zmin, xmax, ymax, zmax = aabb
    return ymin, zmin, ymax - ymin, zmax - zmin


def plot_episode_topdown_and_side(
    states: List[dict],
    title: str = "Episode",
    robot: Optional[object] = None,
    show_arm_proxies: bool = True,
    proxy_stride: int = 5,
    proxy_alpha: float = 0.08,
    # --- debug additions ---
    debug_idx: int = 0,
    print_spheres: bool = True,
    label_spheres: bool = True,
    label_stride: int = 3,
    highlight_debug: bool = True,
    debug_alpha: float = 0.35,
):
    """
    Plots:
      - XY (top-down) with obstacles projected
      - YZ (side) with obstacles projected

    If robot is provided and states contain "q", overlays robot proxy spheres
    using: robot.proxy_spheres(q, ee_radius=state["ee_radius"]) -> List[(center(3,), r)].
    """
    ee = np.array([s["ee_pos"] for s in states], dtype=float)
    goal = np.array(states[0]["goal_pos"], dtype=float)

    obstacles = states[0]["obstacles"]
    bounds = states[0]["bounds"]  # aabb

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # --- XY plot ---
    bxmin, bymin, bzmin, bxmax, bymax, bzmax = bounds
    ax1.set_title("Top-down (x-y)")
    ax1.set_xlim(bxmin, bxmax)
    ax1.set_ylim(bymin, bymax)
    ax1.set_aspect("equal", adjustable="box")

    ax1.add_patch(Rectangle((bxmin, bymin), bxmax - bxmin, bymax - bymin, fill=False))

    for aabb in obstacles:
        x, y, w, h = _project_aabb_xy(aabb)
        ax1.add_patch(Rectangle((x, y), w, h, fill=False))

    ax1.plot(ee[:, 0], ee[:, 1], marker="o", linewidth=1)
    ax1.scatter([ee[0, 0]], [ee[0, 1]], marker="s", label="start")
    ax1.scatter([goal[0]], [goal[1]], marker="*", label="goal")

    # --- YZ plot ---
    ax2.set_title("Side view (y-z)")
    ax2.set_xlim(bymin, bymax)
    ax2.set_ylim(bzmin, bzmax)
    ax2.set_aspect("equal", adjustable="box")

    ax2.add_patch(Rectangle((bymin, bzmin), bymax - bymin, bzmax - bzmin, fill=False))

    for aabb in obstacles:
        y, z, w, h = _project_aabb_yz(aabb)
        ax2.add_patch(Rectangle((y, z), w, h, fill=False))

    ax2.plot(ee[:, 1], ee[:, 2], marker="o", linewidth=1)
    ax2.scatter([ee[0, 1]], [ee[0, 2]], marker="s", label="start")
    ax2.scatter([goal[1]], [goal[2]], marker="*", label="goal")

    # --- Optional: overlay arm proxy spheres ---
    can_draw_proxies = (
        show_arm_proxies
        and (robot is not None)
        and hasattr(robot, "proxy_spheres")
        and ("q" in states[0])
    )
    # Clamp debug idx
    if debug_idx < 0:
        debug_idx = 0
    if debug_idx >= len(states):
        debug_idx = len(states) - 1

    if can_draw_proxies:
        # Draw sampled proxies across trajectory (light)
        for idx in range(0, len(states), max(1, int(proxy_stride))):
            s = states[idx]
            q = np.asarray(s["q"], dtype=float)
            ee_r = float(s.get("ee_radius", 0.035))
            spheres = robot.proxy_spheres(q, ee_radius=ee_r)

            for c, r in spheres:
                c = np.asarray(c, dtype=float)
                ax1.add_patch(Circle((c[0], c[1]), float(r), fill=False, alpha=proxy_alpha))
                ax2.add_patch(Circle((c[1], c[2]), float(r), fill=False, alpha=proxy_alpha))

        # Draw debug frame proxies (strong)
        s_dbg = states[debug_idx]
        q_dbg = np.asarray(s_dbg["q"], dtype=float)
        ee_r_dbg = float(s_dbg.get("ee_radius", 0.035))
        spheres_dbg = robot.proxy_spheres(q_dbg, ee_radius=ee_r_dbg)

        if highlight_debug:
            for i, (c, r) in enumerate(spheres_dbg):
                c = np.asarray(c, dtype=float)
                ax1.add_patch(Circle((c[0], c[1]), float(r), fill=False, alpha=debug_alpha))
                ax2.add_patch(Circle((c[1], c[2]), float(r), fill=False, alpha=debug_alpha))

                if label_spheres and (i % max(1, int(label_stride)) == 0):
                    ax1.text(c[0], c[1], str(i), fontsize=7)
                    ax2.text(c[1], c[2], str(i), fontsize=7)

        # Print sphere list + quick collision hints
        if print_spheres:
            print(f"\n[debug] timestep t={debug_idx} | num_spheres={len(spheres_dbg)}")
            for i, (c, r) in enumerate(spheres_dbg):
                c = np.asarray(c, dtype=float)
                print(f"  sphere[{i:02d}] c={c.round(4)} r={float(r):.4f}")

            # Optional: report which spheres intersect any obstacle AABB
            try:
                from pragmatic_sim_fidelity.simulators.geom3d_sim.collision import sphere_aabb_collides
                hits = []
                for i, (c, r) in enumerate(spheres_dbg):
                    c = np.asarray(c, dtype=float)
                    rr = float(r)
                    for j, obs in enumerate(obstacles):
                        if sphere_aabb_collides(c, rr, np.asarray(obs, dtype=float)):
                            hits.append((i, j))
                if hits:
                    print("[debug] sphere–obstacle hits (sphere_idx, obstacle_idx):", hits)
                else:
                    print("[debug] no sphere–obstacle hits at this timestep")
            except Exception as e:
                print("[debug] (could not check obstacle hits):", repr(e))

        ax1.plot([], [], linestyle="none", marker="o", alpha=proxy_alpha, label="arm proxies")

    ax1.legend(loc="best")
    ax2.legend(loc="best")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()