from __future__ import annotations

from typing import Iterable, Tuple, List
import numpy as np

AABB = Tuple[float, float, float, float, float, float]  # xmin,ymin,zmin,xmax,ymax,zmax


def sphere_aabb_collides(center: np.ndarray, radius: float, aabb: np.ndarray) -> bool:
    """
    center: (3,)
    aabb: (6,) [xmin,ymin,zmin,xmax,ymax,zmax]
    """
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
    for c, r in spheres:
        c = np.asarray(c, dtype=float)
        rr = float(r)
        for obs in obstacles:
            aabb = np.asarray(obs, dtype=float)
            if sphere_aabb_collides(c, rr, aabb):
                return True
    return False


def spheres_out_of_bounds(
    spheres: Iterable[Tuple[np.ndarray, float]],
    bounds: AABB,
) -> bool:
    b = np.asarray(bounds, dtype=float)  # [xmin,ymin,zmin,xmax,ymax,zmax]
    bmin, bmax = b[:3], b[3:]
    for c, r in spheres:
        c = np.asarray(c, dtype=float)
        rr = float(r)
        if np.any(c - rr < bmin) or np.any(c + rr > bmax):
            return True
    return False



def collides_spheres_world(spheres, bounds, obstacles) -> bool:
    # obstacles collide for ALL spheres (full arm)
    if spheres_aabbs_collide(spheres, obstacles):
        return True
    # bounds only for EE sphere (last sphere), with open front
    ee_c, ee_r = spheres[-1]
    return ee_out_of_bounds_open_front(np.asarray(ee_c), float(ee_r), bounds)

def ee_out_of_bounds_open_front(ee: np.ndarray, r: float, bounds: AABB) -> bool:
    """
    Open fridge front: NO collision with y = ymax face.
    Closed faces: xmin, xmax, ymin, zmin, zmax.
    """
    xmin, ymin, zmin, xmax, ymax, zmax = map(float, bounds)
    x, y, z = map(float, ee)
    r = float(r)

    if x - r < xmin: return True
    if x + r > xmax: return True
    if y - r < ymin: return True      # back wall is closed
    # if y + r > ymax: return True     # FRONT OPEN -> do not check
    if z - r < zmin: return True
    if z + r > zmax: return True
    return False