# Import modules for registration side-effects
from .tasks import fridge_reach_ee_3d  # noqa: F401
from .simulators.geom3d_sim import sim as geom3d_sim  # noqa: F401
from .planners import random_shooting  # noqa: F401