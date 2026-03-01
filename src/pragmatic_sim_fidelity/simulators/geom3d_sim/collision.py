from __future__ import annotations
import numpy as np

def sphere_aabb_collides(center: np.ndarray, radius: float, aabb: np.ndarray) -> bool:
    """
    center: (3,)
    aabb: (6,) [xmin,ymin,zmin,xmax,ymax,zmax]
    """
    bmin = aabb[:3]
    bmax = aabb[3:]
    closest = np.minimum(np.maximum(center, bmin), bmax)
    d = center - closest
    return float(d @ d) <= radius * radius