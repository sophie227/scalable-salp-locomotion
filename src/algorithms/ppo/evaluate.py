import torch
from pathlib import Path

from environments.types import EnvironmentParams
from environments.create_env import create_env
from algorithms.ppo.types import Experiment, Params
from algorithms.ppo.ppo import PPO
from algorithms.ppo.utils import get_state_dim, process_state

import dill

from vmas.simulator.utils import save_video


def evaluate(
    exp_config: Experiment,
    env_config: EnvironmentParams,
    device: str,
    trial_id: str,
    dirs: dict,
    disable_subsets: bool = True,
):

    params = Params(**exp_config.params)

    random_seeds = [56, 948, 8137, 6347, 1998]

    # Create environment to get dimension data
    dummy_env = create_env(
        dirs["batch"],
        1,
        device,
        env_config.environment,
        0,
        n_agents=env_config.n_agents,
    )

    d_action = dummy_env.action_space.spaces[0].shape[0]
    d_state = get_state_dim(
        dummy_env.observation_space.spaces[0].shape[0],
        env_config.state_representation,
        exp_config.model,
        env_config.n_agents,
    )

    # get_attention_data(
    #     exp_config,
    #     env_config,
    #     params,
    #     device,
    #     dirs,
    #     env_config.n_agents,
    #     d_state,
    #     d_action,
    # )

    if disable_subsets:
        get_disabled_scalability_data(
            exp_config,
            env_config,
            params,
            device,
            dirs,
            random_seeds[int(trial_id)],
            env_config.n_agents,
            d_state,
            d_action,
        )
    else:
        get_scalability_data(
            exp_config,
            env_config,
            params,
            device,
            dirs,
            random_seeds[int(trial_id)],
            env_config.n_agents,
            d_state,
            d_action,
        )


def get_scalability_data(
    exp_config: Experiment,
    env_config: EnvironmentParams,
    params: Params,
    device: Path,
    dirs: dict,
    # Parameters for scalability experiment
    seed: int,
    n_agents: int,
    d_state: int,
    d_action: int,
    n_rollouts: int = 30,
    extra_agents: int = 40,
):
    n_agents_list = list(range(4, extra_agents + 1, 4))
    data = {n_agents: {} for n_agents in n_agents_list}

    for i, n_agents in enumerate(n_agents_list):

        # Load environment and policy
        env = create_env(
            dirs["batch"],
            n_rollouts,
            device,
            env_config.environment,
            seed,
            training=True,
            n_agents=n_agents,
        )

        learner = PPO(
            device,
            exp_config.model,
            params,
            env_config.n_agents,
            n_agents,
            n_rollouts,
            d_state,
            d_action,
        )
        learner.load(dirs["models"] / "best_model")

        # Set policy to evaluation mode
        learner.policy.eval()

        rewards = []
        distance_rewards = []
        frechet_rewards = []
        episode_count = 0
        state = env.reset()
        cumulative_rewards = torch.zeros(n_rollouts, dtype=torch.float32, device=device)
        cum_dist_rewards = torch.zeros(n_rollouts, dtype=torch.float32, device=device)
        cum_frech_rewards = torch.zeros(n_rollouts, dtype=torch.float32, device=device)
        episode_len = torch.zeros(n_rollouts, dtype=torch.int32, device=device)

        for step in range(0, params.n_max_steps_per_episode):

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
                n_rollouts,
                n_agents,
                d_action,
            ).transpose(1, 0)

            # Turn action tensor into list of tensors with shape (n_env, action_dim)
            action_tensor_list = torch.unbind(action_tensor)

            state, reward, done, info = env.step(action_tensor_list)

            cumulative_rewards += reward[0]
            cum_frech_rewards = info[0]["frechet_rew"]
            cum_dist_rewards = info[0]["distance_rew"]

            episode_len += torch.ones(n_rollouts, dtype=torch.int32, device=device)

            # Create timeout boolean mask
            timeout = episode_len == params.n_max_steps_per_episode

            if torch.any(done) or torch.any(timeout):

                # Get done and timeout indices
                done_indices = torch.nonzero(done).flatten().tolist()
                timeout_indices = torch.nonzero(timeout).flatten().tolist()

                # Merge indices and remove duplicates
                indices = list(set(done_indices + timeout_indices))

                for idx in indices:
                    # Log data when episode is done
                    rewards.append(cumulative_rewards[idx].item())
                    distance_rewards.append(cum_dist_rewards[idx].item())
                    frechet_rewards.append(cum_frech_rewards[idx].item())

                    # Reset vars, and increase counters
                    state = env.reset_at(index=idx)
                    cumulative_rewards[idx] = 0

                    episode_count += 1

            if episode_count >= n_rollouts:
                break

        data[n_agents]["rewards"] = rewards
        data[n_agents]["dist_rewards"] = distance_rewards
        data[n_agents]["frech_rewards"] = frechet_rewards

        print(f"Done evaluating {n_agents}")

    # Store environment
    with open(dirs["logs"] / "evaluation.dat", "wb") as f:
        dill.dump(data, f)


def create_mask(n: int, n_mask: int, device: str):
    """
    Create 4 different masks of size n (must be multiple of 4)

    Mask 1: Alternating 0s and 1s
    Mask 2: Middle half zeros, outer quarters ones
    Mask 3: First half zeros, second half ones
    Mask 4: First half ones, second half zeros
    """
    assert n % 4 == 0, "n must be a multiple of 4"

    match (n_mask):
        case 0:
            # Mask 1: Alternating 0s and 1s
            mask = torch.tensor(
                [i % 2 for i in range(n)], dtype=torch.float32, device=device
            )

        case 1:
            # Mask 2: Middle half zeros (positions n//4 to 3n//4)
            mask = torch.ones(n, dtype=torch.float32, device=device)
            mask[n // 4 : 3 * n // 4] = 0

        case 2:
            # Mask 3: First half zeros, second half ones
            mask = torch.zeros(n, dtype=torch.float32, device=device)
            mask[n // 2 :] = 1

        case 3:
            # Mask 4: First half ones, second half zeros
            mask = torch.ones(n, dtype=torch.float32, device=device)
            mask[n // 2 :] = 0

    return mask


def create_random_mask(n: int, zero_percentage: float, device: str, seed: int = None):
    """
    Create a random mask of size n with a specified percentage of zeros.

    Args:
        n: Size of the mask
        zero_percentage: Percentage of zeros in the mask (between 0 and 1)
        device: Device to place the tensor on
        seed: Optional random seed for reproducibility

    Returns:
        Tensor of size n with ones and zeros
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Calculate number of zeros (round half up)
    n_zeros = int(n * zero_percentage + 0.5)

    # Ensure n_zeros is within valid range
    n_zeros = max(0, min(n, n_zeros))

    # Create mask with all ones
    mask = torch.ones(n, dtype=torch.float32, device=device)

    # Randomly select indices to set to zero
    zero_indices = torch.randperm(n, device=device)[:n_zeros]
    mask[zero_indices] = 0

    return mask

def create_random_mask_by_count(n: int, n_zeros: int, device: str, seed: int = None):
    """
    Create a random mask of size n with a specified number of zeros.

    Args:
        n: Size of the mask
        n_zeros: Number of zeros in the mask
        device: Device to place the tensor on
        seed: Optional random seed for reproducibility

    Returns:
        Tensor of size n with ones and zeros
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Ensure n_zeros is within valid range
    n_zeros = max(0, min(n, n_zeros))

    # Create mask with all ones
    mask = torch.ones(n, dtype=torch.float32, device=device)

    # Randomly select indices to set to zero
    zero_indices = torch.randperm(n, device=device)[:n_zeros]
    mask[zero_indices] = 0

    return mask


def get_disabled_scalability_data(
    exp_config: Experiment,
    env_config: EnvironmentParams,
    params: Params,
    device: Path,
    dirs: dict,
    # Parameters for scalability experiment
    seed: int,
    n_agents: int,
    d_state: int,
    d_action: int,
    n_rollouts: int = 25,
    extra_agents: int = 48,
):
    # n_agents_list = list(range(4, extra_agents + 1, 4))
    # n_agents_list = [int(n_agents * 1.5)]
    n_agents_list = list(range(n_agents//2, int(n_agents * 2.5), int(n_agents * 0.5)))
    
    disabled_subset = False

    data = {n_agents: {} for n_agents in n_agents_list}
        
    for i, n_agents in enumerate(n_agents_list):
        
        n_disabled_masks = n_agents//2

        for n_mask in range(n_disabled_masks):

            # Load environment and policy
            env = create_env(
                dirs["batch"],
                n_rollouts,
                device,
                env_config.environment,
                seed,
                training=True,
                n_agents=n_agents,
            )

            learner = PPO(
                device,
                exp_config.model,
                params,
                env_config.n_agents,
                n_agents,
                n_rollouts,
                d_state,
                2,
            )

            learner.load(dirs["models"] / "best_model")

            # Set policy to evaluation mode
            learner.policy.eval()

            rewards = []
            distance_rewards = []
            frechet_rewards = []
            episode_count = 0
            state = env.reset()
            cumulative_rewards = torch.zeros(
                n_rollouts, dtype=torch.float32, device=device
            )
            cum_dist_rewards = torch.zeros(
                n_rollouts, dtype=torch.float32, device=device
            )
            cum_frech_rewards = torch.zeros(
                n_rollouts, dtype=torch.float32, device=device
            )

            while episode_count < n_rollouts:

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
                    n_rollouts,
                    n_agents,
                    2,
                ).transpose(1, 0)

                # Interpolate to range [0,1]
                action_tensor = (action_tensor + 1) / 2

                # Create (n_agents, n_rollouts, 1) tensor by subtracting second from first
                diff_tensor = (
                    action_tensor[:, :, 0] - action_tensor[:, :, 1]
                ).unsqueeze(-1)

                # Create mask and apply to first dimension
                if disabled_subset:
                    mask_name = n_mask
                    action_mask = create_mask(n_agents, n_mask, device)
                else:
                    mask_name = f"{n_mask+1}"
                    action_mask = create_random_mask_by_count(n_agents, n_mask+1, device)

                diff_tensor = diff_tensor * action_mask.view(n_agents, 1, 1)

                # Turn action tensor into list of tensors with shape (n_env, action_dim)
                action_tensor_list = torch.unbind(diff_tensor)

                state, reward, done, info = env.step(action_tensor_list)

                cumulative_rewards += reward[0]
                cum_frech_rewards = info[0]["frechet_rew"]
                cum_dist_rewards = info[0]["distance_rew"]

                if torch.any(done):

                    # Get done and timeout indices
                    done_indices = torch.nonzero(done).flatten().tolist()

                    # Merge indices and remove duplicates
                    indices = list(set(done_indices))

                    for idx in indices:
                        # Log data when episode is done
                        rewards.append(cumulative_rewards[idx].item())
                        distance_rewards.append(cum_dist_rewards[idx].item())
                        frechet_rewards.append(cum_frech_rewards[idx].item())

                        # Reset vars, and increase counters
                        state = env.reset_at(index=idx)
                        cumulative_rewards[idx] = 0

                        episode_count += 1

            data[n_agents].setdefault(n_mask+1, {})["rewards"] = rewards
            data[n_agents].setdefault(n_mask+1, {})["dist_rewards"] = distance_rewards
            data[n_agents].setdefault(n_mask+1, {})["frech_rewards"] = frechet_rewards

            print(f"Done evaluating {n_agents} agents, mask {mask_name}")

        # Store environment
        with open(
            dirs["logs"] / f"disabled_mask_eval.dat", "wb"
        ) as f:
            dill.dump(data, f)


def get_attention_data(
    exp_config: Experiment,
    env_config: EnvironmentParams,
    params: Params,
    device: str,
    dirs: dict,
    # Parameters for attention experiment
    n_agents: int,
    d_state: int,
    d_action: int,
    extra_agents: int = 64,
    seed=1990,
):

    n_agents_list = list(range(8, extra_agents + 1, 16))
    attention_dict = {}

    for i, n_agents in enumerate(n_agents_list):

        # Load environment
        env = create_env(
            dirs["batch"],
            1,
            device,
            env_config.environment,
            seed,
            training=False,
            n_agents=n_agents,
        )

        # Load PPO agent
        learner = PPO(
            device,
            exp_config.model,
            params,
            env_config.n_agents,
            n_agents,
            1,
            d_state,
            d_action,
        )
        learner.load(dirs["models"] / "best_model")

        # Set policy to evaluation mode
        learner.policy.eval()

        edge_indices = []
        attention_weights = []
        attention_over_time = {
            "Enc_L0": [],  # Encoder self-attention
            "Dec_L0": [],  # Decoder self-attention
            "Cross_L0": [],  # Cross-attention
        }
        match (exp_config.model):
            case (
                "transformer"
                | "transformer_full"
                | "transformer_encoder"
                | "transformer_decoder"
            ):
                attention_weights = learner.policy.build_attention_hooks()

        # Frame list for vide
        frames = []

        # Reset environment
        state = env.reset()

        for _ in range(0, params.n_max_steps_per_episode):

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

            match (exp_config.model):
                case "gat" | "graph_transformer":
                    x = learner.policy.get_batched_graph(
                        process_state(
                            state,
                            env_config.state_representation,
                            exp_config.model,
                        )
                    )
                    _, attention_layers = learner.policy.forward_evaluation(x)

                    # Store edge indices and weights from last layer
                    # Make sure to do a deep copy
                    edge_index, attn_weight = attention_layers[-1]

                    # Store completely detached copies
                    edge_indices.append(edge_index.clone())
                    attention_weights.append(attn_weight.clone())

                case (
                    "transformer"
                    | "transformer_full"
                    | "transformer_encoder"
                    | "transformer_decoder"
                ):
                    _ = learner.policy.forward(
                        process_state(
                            state,
                            env_config.state_representation,
                            exp_config.model,
                        )
                    )

                    # Store attention weights for this timestep
                    for attn_type in attention_over_time:
                        if attn_type in attention_weights:
                            attention_over_time[attn_type].append(
                                attention_weights[attn_type].clone()
                            )

            action_tensor = action.reshape(
                1,
                n_agents,
                d_action,
            ).transpose(1, 0)

            # Turn action tensor into list of tensors with shape (n_env, action_dim)
            action_tensor_list = torch.unbind(action_tensor)

            state, _, done, _ = env.step(action_tensor_list)

            # Store frames for video
            frames.append(
                env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=False,
                )
            )

            if torch.any(done):
                break

        # Save video
        save_video(
            str(dirs["videos"] / f"plots_video_{n_agents}"),
            frames,
            fps=1 / env.scenario.world.dt,
        )

        # Store environment
        attention_dict[n_agents] = {
            "edge_indices": edge_indices,
            "attention_weights": attention_weights,
            "attention_over_time": attention_over_time,
        }

    with open(dirs["logs"] / "attention.dat", "wb") as f:
        dill.dump(attention_dict, f)
