import os
import torch

from environments.types import EnvironmentParams
from environments.create_env import create_env
from algorithms.ppo.types import Experiment, Params
from algorithms.ppo.ppo import PPO
from algorithms.ppo.utils import get_state_dim, process_state

import dill
from pathlib import Path
from statistics import mean
from vmas.simulator.utils import save_video


def view(
    exp_config: Experiment,
    env_config: EnvironmentParams,
    device: str,
    dirs: dict,
    # View parameters
    n_envs=1,
    n_agents_eval=8,
    n_rollouts=10,
    rollout_length=512,
    seed=1025,
    render=True,
):

    params = Params(**exp_config.params)

    n_agents_train = env_config.n_agents

    env = create_env(
        dirs["batch"],
        n_envs,
        device,
        env_config.environment,
        seed,  # 10265
        n_agents=n_agents_eval,
        training=False,
    )

    d_action = env.action_space.spaces[0].shape[0]
    d_state = get_state_dim(
        env.observation_space.spaces[0].shape[0],
        env_config.state_representation,
        exp_config.model,
        n_agents_train,
    )

    learner = PPO(
        device,
        exp_config.model,
        params,
        n_agents_train,
        n_agents_eval,
        n_envs,
        d_state,
        d_action,
    )
    learner.load(dirs["models"] / "checkpoint")
    learner.policy.eval()

    frame_list = []
    info_list = []
    total_rew_per_rollout = []

    for i in range(n_rollouts):

        done = False
        state = env.reset()
        rew = 0

        for t in range(0, rollout_length):

            action = torch.clamp(
                learner.deterministic_action(
                    process_state(
                        state,
                        env_config.state_representation,
                        exp_config.model,
                    )
                ),
                min=-1.0,
                max=1.0,
            )

            action_tensor = action.reshape(
                n_envs,
                n_agents_eval,
                d_action,
            ).transpose(1, 0)

            # Turn action tensor into list of tensors with shape (n_env, action_dim)
            action_tensor_list = torch.unbind(action_tensor)

            state, reward, done, info = env.step(action_tensor_list)

            info_list.append(info[0])

            rew += reward[0].item()

            frame = env.render(
                mode="rgb_array",
                agent_index_focus=None,  # Can give the camera an agent index to focus on
                visualize_when_rgb=render,
            )

            frame_list.append(frame)

            if torch.any(done):
                print("DONE")
                break

        total_rew_per_rollout.append(rew)

        print(f"TOTAL RETURN: {rew}")

    print(f"MEAN RETURN OVER {n_rollouts}: {mean(total_rew_per_rollout)}")

    with open(dirs["logs"] / f"test_rollouts_info_{n_agents_eval}.dat", "wb") as f:
        dill.dump(info_list, f)

    save_video(
        str(dirs["videos"] / f"view_{n_agents_eval}"),
        frame_list,
        fps=1 / env.scenario.world.dt,
    )
