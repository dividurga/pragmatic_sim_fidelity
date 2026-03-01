from __future__ import annotations
import numpy as np
from typing import List
from ..core.types import Action, State
from ..core.task import TaskSpec


def sample_move_ee_sequence_goal_biased(
    rng,
    task: TaskSpec,
    state: State,
    horizon: int,
) -> List[Action]:
    """
    Goal-biased sampler for move_ee primitive.
    Works for any task that has:
        state["ee_pos"]
        state["goal_pos"]
        state["step_size"]
    """

    step = float(state.get("step_size", 0.05))
    ee = np.array(state["ee_pos"], dtype=float)
    goal = np.array(state["goal_pos"], dtype=float)

    seq: List[Action] = []

    for _ in range(horizon):
        direction = goal - ee
        norm = np.linalg.norm(direction) + 1e-9
        direction = direction / norm

        # Bias toward goal + small noise
        delta = direction * step + rng.normal(0.0, step * 0.3, size=(3,))
        delta = np.clip(delta, -step, step)

        seq.append(
            Action(
                kind="move_ee",
                params={
                    "dx": float(delta[0]),
                    "dy": float(delta[1]),
                    "dz": float(delta[2]),
                },
            )
        )

        ee = ee + delta  # internal rollout of sampled sequence

    return seq