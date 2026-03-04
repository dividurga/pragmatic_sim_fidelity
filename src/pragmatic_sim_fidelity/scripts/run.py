"""Run an episode of a task with a given planner and sims, and visualize the results."""

# need to fix the modularity of this script,
# it's currently importing a lot of # internal details from the core and
# samplers
from __future__ import annotations

import argparse

import numpy as np

import pragmatic_sim_fidelity.simulators.geom3d_sim.collision  # pylint: disable=unused-import

# Import modules for registration side effects.
import pragmatic_sim_fidelity.simulators.geom3d_sim.sim  # pylint: disable=unused-import
import pragmatic_sim_fidelity.tasks.fridge_reach_ee_3d  # pylint: disable=unused-import
from pragmatic_sim_fidelity.core.pipeline import run_episode
from pragmatic_sim_fidelity.planners.random_shooting import RandomShooting
from pragmatic_sim_fidelity.registry import PLANNERS, SIMS, TASKS
from pragmatic_sim_fidelity.samplers.delta_q import sample_delta_q_sequence_gaussian
from pragmatic_sim_fidelity.viz.geom3d_debug import (
    make_panda7_fk_fn_to_hand,
    plot_episode_xy_yz,
)


def main():
    """Run an episode."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="fridge_reach_franka_geom3d")
    ap.add_argument("--plan_sim", default="geom3d")
    ap.add_argument("--exec_sim", default="geom3d")
    ap.add_argument("--planner", default="random_shooting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--samples", type=int, default=100)
    ap.add_argument("--sigma", type=float, default=0.3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    task = TASKS[args.task]()
    plan_sim = SIMS[args.plan_sim]()
    exec_sim = SIMS[args.exec_sim]()

    planner = PLANNERS[args.planner]()
    if isinstance(planner, RandomShooting):
        planner.horizon = args.horizon
        planner.num_samples = args.samples
        planner.sampler = (
            lambda rng_, task_, state_, horizon_: sample_delta_q_sequence_gaussian(
                rng=rng_,
                horizon=horizon_,
                sigma=args.sigma,
            )
        )

    result = run_episode(
        task, planner, plan_sim, exec_sim, rng, max_steps=500, mpc_execute_k=1
    )

    states = result.states

    p = getattr(task, "p", None) or getattr(exec_sim, "p", None)
    franka_id = (
        getattr(task, "robot_id", None)
        or getattr(task, "franka_id", None)
        or getattr(exec_sim, "robot_id", None)
        or getattr(exec_sim, "franka_id", None)
    )

    if p is not None and franka_id is not None and "q" in states[0]:
        fk_fn, joint_idxs, link_chain = make_panda7_fk_fn_to_hand(
            p, franka_id, hand_link_name="panda_hand"
        )
        print("joint_idxs:", joint_idxs)
        print("link_chain:", link_chain)

        plot_episode_xy_yz(
            states,
            title="Episode w/ Panda overlay",
            fk_points_fn=fk_fn,
            arm_stride=6,
        )
    else:
        # Fall back to EE-only plot if we don't have bullet handles here
        plot_episode_xy_yz(states, title="Episode")

    print(f"success={result.success} steps={result.steps}")
    print("final ee_pos:", result.states[-1]["ee_pos"])
    print("goal_pos:", result.states[-1]["goal_pos"])
    print("collided:", result.states[-1].get("collided", False))


if __name__ == "__main__":
    main()
