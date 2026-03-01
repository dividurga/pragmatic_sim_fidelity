
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
AABB = Tuple[float, float, float, float, float, float]

def _project_aabb_xy(aabb: AABB):
    xmin, ymin, zmin, xmax, ymax, zmax = aabb
    return xmin, ymin, xmax - xmin, ymax - ymin

def _project_aabb_yz(aabb: AABB):
    xmin, ymin, zmin, xmax, ymax, zmax = aabb
    return ymin, zmin, ymax - ymin, zmax - zmin

def plot_episode_topdown_and_side(states: List[dict], title: str = "Episode"):
    """
    Plots:
      - XY (top-down) with obstacles projected
      - YZ (side) with obstacles projected
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

    # bounds rectangle
    ax1.add_patch(Rectangle((bxmin, bymin), bxmax - bxmin, bymax - bymin, fill=False))

    # obstacles projected to XY
    for aabb in obstacles:
        x, y, w, h = _project_aabb_xy(aabb)
        ax1.add_patch(Rectangle((x, y), w, h, fill=False))

    ax1.plot(ee[:, 0], ee[:, 1], marker="o", linewidth=1)
    ax1.scatter([ee[0, 0]], [ee[0, 1]], marker="s", label="start")
    ax1.scatter([goal[0]], [goal[1]], marker="*", label="goal")
    ax1.legend(loc="best")

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
    ax2.legend(loc="best")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()