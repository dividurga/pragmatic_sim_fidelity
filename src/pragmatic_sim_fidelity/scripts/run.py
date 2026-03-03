from __future__ import annotations
import argparse
import numpy as np

from pragmatic_sim_fidelity import plugins  # noqa: F401
from pragmatic_sim_fidelity.registry import TASKS, SIMS, PLANNERS
from pragmatic_sim_fidelity.core.pipeline import run_episode
from pragmatic_sim_fidelity.planners.random_shooting import RandomShooting
from pragmatic_sim_fidelity.samplers.delta_q import sample_delta_q_sequence_gaussian


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="fridge_reach_franka_geom3d")
    ap.add_argument("--plan_sim", default="geom3d")
    ap.add_argument("--exec_sim", default="geom3d")
    ap.add_argument("--planner", default="random_shooting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--samples", type=int, default=512)
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
        planner.sampler = lambda rng_, task_, state_, horizon_: sample_delta_q_sequence_gaussian(
            rng=rng_,
            task=task_,
            state=state_,
            horizon=horizon_,
            sigma=args.sigma,
        )

    result = run_episode(task, planner, plan_sim, exec_sim, rng, max_steps=3, mpc_execute_k=1)

    from pragmatic_sim_fidelity.viz.geom3d_debug import plot_episode_topdown_and_side
   
    # after running episode:
    plot_episode_topdown_and_side(
    result.states,
    title="debug",
    robot=getattr(task, "_robot", None),
    debug_idx=0,          # show spheres at t=0
    print_spheres=True,
    label_spheres=True,
    label_stride=2,
)

    print(f"success={result.success} steps={result.steps}")
    print("final ee_pos:", result.states[-1]["ee_pos"])
    print("goal_pos:", result.states[-1]["goal_pos"])
    print("collided:", result.states[-1].get("collided", False))


if __name__ == "__main__":
    main()