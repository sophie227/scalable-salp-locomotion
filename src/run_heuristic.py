#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import time
import argparse
import yaml
import torch

from vmas.simulator.utils import save_video
from vmas.simulator.environment import Environment

from environments.create_env import create_env
from environments.types import EnvironmentEnum
from pynput.keyboard import Listener
from testing.manual_control import manual_control
from pathlib import Path


def use_vmas_env(
    name: str,
    batch_dir: Path,
    env_name: str,
    seed: int,
    env: Environment = None,
    render: bool = False,
    save_render: bool = False,
    n_envs: int = 1,
    n_steps: int = 100,
    n_agents: int = 8,
    device: str = "cpu",
    rotating_salps: bool = True,
    visualize_render: bool = True,
):
    """Example function to use a vmas environment

    Args:
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        save_render (bool):  Whether to save render of the scenario
        n_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action
        visualize_render (bool, optional): Whether to visualize the render. Defaults to ``True``.
        kwargs (dict, optional): Keyword arguments to pass to the scenario

    Returns:

    """
    assert not (save_render and not render), "To save the video you have to render it"

    frame_list = []  # For creating a gif
    init_time = time.time()
    mc = manual_control(n_agents)

    if env is None:
        env = create_env(
            batch_dir=batch_dir,
            n_envs=n_envs,
            device=device,
            env_name=env_name,
            seed=seed,
            n_agents=n_agents,
            training=False,
            rotating_salps=rotating_salps,
        )

    _ = env.reset()

    with Listener(on_press=mc.on_press, on_release=mc.on_release) as listener:
        listener.join(timeout=0.1)

        for _ in range(n_steps):

            actions = []
            cmd_action = torch.tensor(mc.cmd_vel, dtype=torch.float32, device=device)
            cmd_action = cmd_action.repeat(n_envs, 1)

            for _agent in env.agents:
                action = cmd_action.clone()
                actions.append(action)

            env.step(actions)

            if render:
                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=visualize_render,
                )
                if save_render:
                    frame_list.append(frame)

    total_time = time.time() - init_time

    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device} "
        f"for {name} scenario."
    )

    if render and save_render:
        save_video(name, frame_list, fps=10 / env.scenario.world.dt)


if __name__ == "__main__":
    # Arg parser variables
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch",
        default="",
        help="Experiment batch",
        type=str,
    )
    parser.add_argument(
        "--name",
        default="",
        help="Experiment name",
        type=str,
    )
    parser.add_argument(
        "--environment",
        default=EnvironmentEnum.VMAS_SALP_PASSAGE_CURR,
        help="Learning environment name",
        type=str,
    )

    parser.add_argument("--trial_id", default=0, help="Sets trial ID", type=int)
    parser.add_argument("--n_agents", default=8, type=int)
    parser.add_argument("--n_steps", default=1000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--rotating_salps",
        action="store_true",
        help="Enable turning with left/right keys.",
    )

    args = vars(parser.parse_args())

    # Set base_config path
    dir_path = Path(__file__).parent

    # Set configuration folder
    batch_dir = dir_path / "experiments" / "yamls" / args["batch"]

    env_file = batch_dir / "_env.yaml"

    with open(str(env_file), "r") as file:
        env_config = yaml.safe_load(file)

    n_envs = 1

    use_vmas_env(
        name=f"{args['batch']}_{args['n_agents']}a",
        batch_dir=batch_dir,
        env_name=args["environment"],
        seed=args["seed"],
        render=True,
        save_render=False,
        device="cpu",
        n_envs=n_envs,
        n_steps=args["n_steps"],
        n_agents=args["n_agents"],
        rotating_salps=args["rotating_salps"],
    )
