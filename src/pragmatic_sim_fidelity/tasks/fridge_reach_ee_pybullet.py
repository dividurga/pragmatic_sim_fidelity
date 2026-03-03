from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional
import os
import numpy as np
import pybullet as p
import pybullet_data

AABB = Tuple[float, float, float, float, float, float]  # xmin,ymin,zmin,xmax,ymax,zmax


def aabb_center_halfextents(aabb: AABB):
    xmin, ymin, zmin, xmax, ymax, zmax = map(float, aabb)
    cx, cy, cz = (xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5
    hx, hy, hz = (xmax - xmin) * 0.5, (ymax - ymin) * 0.5, (zmax - zmin) * 0.5
    return (cx, cy, cz), (hx, hy, hz)


def try_load_urdf(path: str, base_pos, base_orn=(0, 0, 0, 1), fixed_base=True, global_scaling=1.0):
    """Tries to load a URDF; returns body id or None if load fails."""
    try:
        bid = p.loadURDF(
            path,
            basePosition=base_pos,
            baseOrientation=base_orn,
            useFixedBase=fixed_base,
            globalScaling=global_scaling,
        )
        return bid
    except Exception as e:
        print(f"[warn] failed to load URDF '{path}': {e}")
        return None


def create_static_box_from_aabb(aabb: AABB, rgba=(0.7, 0.7, 0.7, 1.0), friction=0.8):
    center, half = aabb_center_halfextents(aabb)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    bid = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=center)
    p.changeDynamics(bid, -1, lateralFriction=friction)
    return bid


def create_static_cylinder_approx_from_aabb(aabb: AABB, rgba=(0.2, 0.6, 0.9, 1.0), friction=0.6):
    """
    Good "bottle-ish" approximation:
    - radius ~ half of max(x,y) span
    - height ~ z span
    """
    (cx, cy, cz), (hx, hy, hz) = aabb_center_halfextents(aabb)
    radius = float(max(hx, hy))
    height = float(2.0 * hz)
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=height, rgbaColor=rgba)
    bid = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=(cx, cy, cz))
    p.changeDynamics(bid, -1, lateralFriction=friction)
    return bid


def create_static_egg_approx_from_aabb(aabb: AABB, rgba=(0.95, 0.9, 0.7, 1.0), friction=0.5):
    """
    Egg-ish approximation using a capsule.
    """
    (cx, cy, cz), (hx, hy, hz) = aabb_center_halfextents(aabb)
    # Capsule in Bullet is along Z; choose radius from x/y, height from z minus hemispheres.
    radius = float(max(hx, hy))
    total_height = float(2.0 * hz)
    cyl_height = max(0.0, total_height - 2.0 * radius)
    col = p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=cyl_height)
    vis = p.createVisualShape(p.GEOM_CAPSULE, radius=radius, length=cyl_height, rgbaColor=rgba)
    bid = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=(cx, cy, cz))
    p.changeDynamics(bid, -1, lateralFriction=friction)
    return bid


@dataclass
class FridgeSceneSpec:
    # Same as your reset()
    bounds: AABB = (0.0, 0.0, 0.0, 0.8, 0.6, 0.8)

    shelf_slab: AABB = (0.0, 0.0, 0.35, 0.80, 0.6, 0.38)
    bottle1_aabb: AABB = (0.35, 0.22, 0.38, 0.43, 0.30, 0.65)
    bottle2_aabb: AABB = (0.62, 0.45, 0.38, 0.70, 0.50, 0.50)
    egg_aabb: AABB = (0.22, 0.33, 0.38, 0.26, 0.37, 0.50)

    # Optional URDF asset paths (either absolute, or relative to pybullet_data / your project)
    bottle_urdf: Optional[str] = None
    egg_urdf: Optional[str] = None

    # If you have mesh-only assets, easiest route is: make URDFs. (I can give a URDF template if you want.)
    bottle_scale: float = 1.0
    egg_scale: float = 1.0


def build_fridge_scene(spec: FridgeSceneSpec, gui=True):
    """
    Builds the PyBullet scene using the same geometry as the AABB setup.
    - shelf is a static box
    - bottles/egg: tries URDF if provided, else uses cylinder/capsule approximations from AABBs
    """
    cid = p.connect(p.GUI if gui else p.DIRECT)
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)

    # Search paths: pybullet_data + cwd
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(os.getcwd())

    p.loadURDF("plane.urdf", basePosition=(0, 0, -1.0), useFixedBase=True)

    # Draw workspace bounds as debug lines (visual only)
    xmin, ymin, zmin, xmax, ymax, zmax = spec.bounds
    corners = [
        (xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin),
        (xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax),
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for i, j in edges:
        p.addUserDebugLine(corners[i], corners[j], lineColorRGB=(0.2, 0.2, 0.2), lineWidth=1.0)

    # Shelf slab (static)
    shelf_id = create_static_box_from_aabb(spec.shelf_slab, rgba=(0.55, 0.55, 0.6, 1.0), friction=0.9)

    # Bottles + egg
    created = {"shelf": shelf_id, "bottle1": None, "bottle2": None, "egg": None}

    # Bottle 1
    b1_center, _ = aabb_center_halfextents(spec.bottle1_aabb)
    if spec.bottle_urdf:
        b1 = try_load_urdf(spec.bottle_urdf, base_pos=b1_center, fixed_base=True, global_scaling=spec.bottle_scale)
        created["bottle1"] = b1 if b1 is not None else create_static_cylinder_approx_from_aabb(spec.bottle1_aabb)
    else:
        created["bottle1"] = create_static_cylinder_approx_from_aabb(spec.bottle1_aabb)

    # Bottle 2
    b2_center, _ = aabb_center_halfextents(spec.bottle2_aabb)
    if spec.bottle_urdf:
        b2 = try_load_urdf(spec.bottle_urdf, base_pos=b2_center, fixed_base=True, global_scaling=spec.bottle_scale)
        created["bottle2"] = b2 if b2 is not None else create_static_cylinder_approx_from_aabb(
            spec.bottle2_aabb, rgba=(0.15, 0.5, 0.8, 1.0)
        )
    else:
        created["bottle2"] = create_static_cylinder_approx_from_aabb(spec.bottle2_aabb, rgba=(0.15, 0.5, 0.8, 1.0))

    # Egg
    egg_center, _ = aabb_center_halfextents(spec.egg_aabb)
    if spec.egg_urdf:
        e = try_load_urdf(spec.egg_urdf, base_pos=egg_center, fixed_base=True, global_scaling=spec.egg_scale)
        created["egg"] = e if e is not None else create_static_egg_approx_from_aabb(spec.egg_aabb)
    else:
        created["egg"] = create_static_egg_approx_from_aabb(spec.egg_aabb)

    # Helpful camera
    p.resetDebugVisualizerCamera(
        cameraDistance=1.1,
        cameraYaw=35,
        cameraPitch=-25,
        cameraTargetPosition=(0.4, 0.3, 0.45),
    )

    return cid, created


if __name__ == "__main__":
    # 1) Box/cylinder/capsule approximations (runs immediately)
    spec = FridgeSceneSpec()

    # 2) When you have assets, set these to your URDF paths, e.g.
    # spec.bottle_urdf = "assets/bottle.urdf"
    # spec.egg_urdf = "assets/egg.urdf"
    # spec.bottle_scale = 1.0
    # spec.egg_scale = 1.0

    cid, ids = build_fridge_scene(spec, gui=True)
    print(ids)

    # keep sim alive if GUI
    while p.isConnected():
        p.stepSimulation()