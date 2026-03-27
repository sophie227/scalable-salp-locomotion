import os
from pathlib import Path
import yaml

from vmas import make_env
from vmas.simulator.environment import Environment

from environments.rover.rover_domain import RoverDomain
from environments.salp_navigate.domain import SalpNavigateDomain
from environments.salp_navigate_lidar.domain import SalpNavigateLidarDomain
from environments.salp_passage.domain import SalpPassageDomain

from environments.types import EnvironmentEnum


def create_vmas_env(n_envs, device, seed, env_args):
    env = make_env(
        num_envs=n_envs,
        device=device,
        seed=seed,
        # Environment specific variables
        **env_args,
    )
    return env


def create_env(
    batch_dir,
    n_envs: int,
    device: str,
    env_name: str,
    seed: int,
    **kwargs,
) -> Environment:

    env_file = os.path.join(batch_dir, "_env.yaml")

    with open(str(env_file), "r") as file:
        env_config = yaml.safe_load(file)

    match (env_name):
        case EnvironmentEnum.VMAS_BUZZ_WIRE:
            # Environment arguments
            env_args = {
                # Environment data
                "scenario": "buzz_wire",
            }
            return create_vmas_env(n_envs, device, seed, env_args)

        case EnvironmentEnum.VMAS_BALANCE:
            # Environment arguments
            env_args = {
                # Environment data
                "scenario": "balance",
            }
            return create_vmas_env(n_envs, device, seed, env_args)

        case EnvironmentEnum.VMAS_ROVER:

            # Environment arguments
            env_args = {
                # Environment data
                "scenario": RoverDomain(),
                "x_semidim": env_config["map_size"][0],
                "y_semidim": env_config["map_size"][1],
                # Agent data
                "n_agents": len(env_config["agents"]),
                "agents_colors": [
                    agent["color"] if agent.get("color") else "BLUE"
                    for agent in env_config["agents"]
                ],
                "agents_positions": [
                    poi["position"]["coordinates"] for poi in env_config["agents"]
                ],
                "lidar_range": [
                    rover["observation_radius"] for rover in env_config["agents"]
                ][0],
                # POIs data
                "n_targets": len(env_config["targets"]),
                "targets_positions": [
                    poi["position"]["coordinates"] for poi in env_config["targets"]
                ],
                "targets_values": [poi["value"] for poi in env_config["targets"]],
                "targets_types": [poi["type"] for poi in env_config["targets"]],
                "targets_orders": [poi["order"] for poi in env_config["targets"]],
                "targets_colors": [
                    poi["color"] if poi.get("color") else "GREEN"
                    for poi in env_config["targets"]
                ],
                "agents_per_target": [poi["coupling"] for poi in env_config["targets"]][
                    0
                ],
                "covering_range": [
                    poi["observation_radius"] for poi in env_config["targets"]
                ][0],
                "use_order": env_config["use_order"],
                "viewer_zoom": kwargs.pop("viewer_zoom", 1),
            }
            return create_vmas_env(n_envs, device, seed, env_args)

        case EnvironmentEnum.VMAS_SALP_NAVIGATE:
            env_args = {
                # Environment data
                "scenario": SalpNavigateDomain(),
                "training": kwargs.get("training", True),
                # Agent data
                "n_agents": kwargs.get("n_agents", 1),
                "state_representation": env_config["state_representation"],
                "rotating_salps": env_config["rotating_salps"],
            }
            return create_vmas_env(n_envs, device, seed, env_args)

        case EnvironmentEnum.VMAS_SALP_NAVIGATE_LIDAR:
            env_args = {
                # Environment data
                "scenario": SalpNavigateLidarDomain(),
                "training": kwargs.get("training", True),
                # Agent data
                "n_agents": kwargs.get("n_agents", 1),
                "state_representation": env_config.get("state_representation", "local"),
                "rotating_salps": env_config.get("rotating_salps", False),
            }
            return create_vmas_env(n_envs, device, seed, env_args)

        

        case EnvironmentEnum.VMAS_SALP_PASSAGE:
            env_args = {
                # Environment data
                "scenario": SalpPassageDomain(),
                "training": kwargs.get("training", True),
                # Agent data
                "n_agents": kwargs.get("n_agents", 1),
                "state_representation": env_config["state_representation"],
            }
            return create_vmas_env(n_envs, device, seed, env_args)

        case EnvironmentEnum.MAMUJOCO_SWIMMER:
            # TODO: add actual mamujoco env code
            env_args = {}
            return None
