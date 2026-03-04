"""Collision checking utilities for simple 3D geometry-based simulator."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np

AABB = Tuple[float, float, float, float, float, float]  # xmin,ymin,zmin,xmax,ymax,zmax


def collided_obstacles_only(
    spheres: List[Tuple[np.ndarray, float]],
    obstacles: List[AABB],
) -> bool:
    """Check if any Franka sphere collides with any AABB obstacle.""" """"""
    return spheres_aabbs_collide(spheres, obstacles)


def sphere_aabb_collides(center: np.ndarray, radius: float, aabb: np.ndarray) -> bool:
    """Checks if a sphere with given center and radius collides with an axis-aligned
    bounding box (AABB)."""
    center = np.asarray(center, dtype=float)
    aabb = np.asarray(aabb, dtype=float)

    bmin = aabb[:3]
    bmax = aabb[3:]
    closest = np.minimum(np.maximum(center, bmin), bmax)
    d = center - closest
    return float(d @ d) <= float(radius) * float(radius)


def spheres_aabbs_collide(
    spheres: Iterable[Tuple[np.ndarray, float]],
    obstacles: List[AABB],
) -> bool:
    """Check if any sphere collides with any AABB obstacle."""
    for c, r in spheres:
        c = np.asarray(c, dtype=float)
        rr = float(r)
        for obs in obstacles:
            aabb = np.asarray(obs, dtype=float)
            if sphere_aabb_collides(c, rr, aabb):
                return True
    return False
