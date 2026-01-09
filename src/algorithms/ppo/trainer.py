import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from environments.types import EnvironmentParams
from environments.create_env import create_env
from algorithms.ppo.types import Experiment, Params
from algorithms.ppo.ppo import PPO
from algorithms.ppo.utils import get_state_dim, process_state

import dill
import random
import time


class PPOTrainer:

    def __init__(
        self,
        exp_config: Experiment,
        env_config: EnvironmentParams,
        device: str,
        trial_id: str,
        dirs: dict,
        checkpoint: bool = False,
    ):
        self.params = Params(**exp_config.params)

        # Create environment
        self.n_envs = env_config.n_envs
        self.n_agents = env_config.n_agents
        self.env_name = env_config.environment
        self.state_representation = env_config.state_representation
        self.model = exp_config.model

        self.device = device

        self.trial_id = trial_id

        self.dirs = dirs

        self.checkpoint = checkpoint

    def train(
        self,
    ):

        # Set seeds
        random_seed = self.params.random_seeds[0]

        if self.trial_id.isdigit():
            random_seed = self.params.random_seeds[int(self.trial_id)]

        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)

        env = create_env(
            self.dirs["batch"],
            self.n_envs,
            n_agents=self.n_agents,
            device=self.device,
            env_name=self.env_name,
            seed=random_seed,
        )

        # Set state and action dimensions
        self.d_action = env.action_space.spaces[0].shape[0]
        d_state = get_state_dim(
            env.observation_space.spaces[0].shape[0],
            self.state_representation,
            self.model,
            self.n_agents,
        )

        # Create learner object
        self.learner = PPO(
            self.device,
            self.model,
            self.params,
            self.n_agents,
            self.n_agents,
            self.n_envs,
            d_state,
            self.d_action,
            self.checkpoint,
        )

        # Create training data logging object
        training_data = {
            "steps": [],
            "dones": [],
            "episodes": [],
            "timestamps": [],
            "rewards_per_iteration": [],
            "reset_count": 1,
            "last_step_count": 0,
            "last_episode_count": 0,
            "best_reward": -1e1000,
        }

        # Checkpoint loading logic
        if self.checkpoint:

            checkpoint_path = self.dirs["models"] / "checkpoint"

            if checkpoint_path.is_file():
                # Load checkpoint
                self.learner.load(checkpoint_path)

                # Load data
                with open(self.dirs["logs"] / "train.dat", "rb") as data_file:
                    training_data = dill.load(data_file)

                # Load env up to checkpoint
                # with open(dirs["models"] / "env.dat", "rb") as env_file:
                #     env = dill.load(env_file)

        # Setup loop variables
        global_step = training_data["last_step_count"]
        total_episodes = training_data["last_episode_count"]
        checkpoint_step = 0

        # Check if we are done before starting
        if global_step >= self.params.n_total_steps:
            print(f"Training was already finished at {global_step} steps")
            return

        while global_step < self.params.n_total_steps:

            # Log start time
            start_time = time.time()

            episode_len = torch.zeros(
                self.n_envs, dtype=torch.int32, device=self.device
            )
            cum_rewards = torch.zeros(
                self.n_envs, dtype=torch.float32, device=self.device
            )

            # Move random seed to checkpoint by calling reset multiple times
            for _ in range(training_data["reset_count"]):
                state = env.reset()

            # Collect batch of data stepping by n_envs
            for _ in range(0, self.params.batch_size, self.n_envs):

                # Clamp because actions are stochastic and can lead to them been out of -1 to 1 range
                b_state, b_action, b_logprob, b_state_val = self.learner.select_action(
                    process_state(
                        state,
                        self.state_representation,
                        self.model,
                    )
                )
                actions_per_env = torch.clamp(
                    b_action,
                    min=-1.0,
                    max=1.0,
                )

                # Permute action tensor of shape (n_envs, n_agents * action_dim) to (agents, n_env, action_dim)
                action_tensor = actions_per_env.view(
                    self.n_envs, self.n_agents, self.d_action
                )

                # Turn action tensor into list of tensors with shape (n_env, action_dim)
                action_tensor_list = [action_tensor[:, i] for i in range(self.n_agents)]

                state, reward, done, _ = env.step(action_tensor_list)

                # Add data to learner buffer
                self.learner.buffer.add(
                    b_state, b_action, b_logprob, b_state_val, reward[0], done
                )

                cum_rewards += reward[0]

                episode_len += torch.ones(
                    self.n_envs, dtype=torch.int32, device=self.device
                )

                # Increase counters
                global_step += self.n_envs

                if torch.any(done):

                    # Get done and timeout indices
                    done_indices = torch.nonzero(done).flatten().tolist()

                    # Merge indices and remove duplicates
                    indices = list(set(done_indices))

                    for idx in indices:
                        training_data["reset_count"] += 1

                        # Reset vars, and increase counters
                        state = env.reset_at(index=idx)
                        cum_rewards[idx], episode_len[idx] = 0, 0

                        total_episodes += 1

                    training_data["dones"].append(len(indices))
                    training_data["steps"].append(global_step)
                    training_data["episodes"].append(total_episodes)
                    training_data["timestamps"].append(time.time() - start_time)
                    training_data["last_step_count"] = global_step
                    training_data["last_episode_count"] = total_episodes

            # Do training step
            self.learner.update()

            # Evaluate model after training iteration
            training_data["rewards_per_iteration"].append(self.evaluate_model())

            # Store model if we get a new best reward
            if (
                training_data["rewards_per_iteration"][-1]
                > training_data["best_reward"]
            ):
                self.learner.save(self.dirs["models"] / "best_model")
                training_data["best_reward"] = training_data["rewards_per_iteration"][
                    -1
                ]

            # Store full checkpoint
            if global_step - checkpoint_step >= 10000:

                # Save model
                self.learner.save(self.dirs["models"] / "checkpoint")

                # Store reward per episode data
                with open(self.dirs["logs"] / "train.dat", "wb") as f:
                    dill.dump(training_data, f)

                # Store environment
                # with open(self.dirs["models"] / "env.dat", "wb") as f:
                #     dill.dump(env, f)

                checkpoint_step = global_step

                if global_step >= self.params.n_total_steps:
                    print("Finished training")
                    break

                print(
                    f"Step: {global_step}, Episodes: {total_episodes}, Best Reward: {training_data["best_reward"]}, Latest Eval: {training_data["rewards_per_iteration"][-1]}, Minutes {'{:.2f}'.format((sum(training_data['timestamps'])) / 60)}"
                )

    def evaluate_model(self, evaluations=5):
        # Set policy to evaluation mode
        self.learner.policy.eval()

        env = create_env(
            self.dirs["batch"],
            evaluations,
            n_agents=self.n_agents,
            device=self.device,
            env_name=self.env_name,
            seed=random.randint(0, 10000),
        )

        done = torch.zeros((evaluations), device=self.device)
        done_mask = torch.ones(
            (evaluations), device=self.device
        )  # Once we are done in a particular environment stop cumulating rewards for that environment
        state = env.reset()

        cum_rewards = torch.zeros(evaluations, dtype=torch.float32, device=self.device)
        step = 0
        while not torch.all(done_mask == 0):

            b_action = self.learner.deterministic_action(
                process_state(
                    state,
                    self.state_representation,
                    self.model,
                )
            )

            actions_per_env = torch.clamp(
                b_action,
                min=-1.0,
                max=1.0,
            )

            # Permute action tensor of shape (n_envs, n_agents * action_dim) to (agents, n_env, action_dim)
            action_tensor = actions_per_env.view(
                evaluations, self.n_agents, self.d_action
            )

            # Turn action tensor into list of tensors with shape (n_env, action_dim)
            action_tensor_list = [action_tensor[:, i] for i in range(self.n_agents)]

            state, reward, done, _ = env.step(action_tensor_list)

            cum_rewards += reward[0] * done_mask

            # Update done mask
            done_indices = torch.nonzero(done).flatten().tolist()

            done_mask[done_indices] = 0

            step += 1

        self.learner.policy.train()

        return cum_rewards.mean().item()
