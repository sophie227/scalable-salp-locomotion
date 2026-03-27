import os
import yaml
import torch
from pathlib import Path

from algorithms.ppo.run import PPO_Runner
from algorithms.ppo.types import Experiment as PPO_Experiment

# from algorithms.manual.control import ManualControl

from algorithms.types import AlgorithmEnum

from environments.types import EnvironmentEnum, EnvironmentParams
from environments.rover.types import RoverEnvironmentParams
from environments.salp_navigate.types import SalpNavigateEnvironmentParams
from environments.salp_passage.types import SalpPassageEnvironmentParams


def run_algorithm(
    batch_dir: Path,
    batch_name: str,
    experiment_name: str,
    algorithm: str,
    environment: str,
    trial_id: str,
    view: bool = False,
    checkpoint: bool = False,
    evaluate: bool = False,
):

    # Load environment config
    env_file = batch_dir / "_env.yaml"

    with open(env_file, "r") as file:
        env_dict = yaml.safe_load(file)

    match (environment):
        case EnvironmentEnum.VMAS_ROVER:
            env_config = RoverEnvironmentParams(**env_dict)

        case EnvironmentEnum.VMAS_SALP_NAVIGATE:
            env_config = SalpNavigateEnvironmentParams(**env_dict)

        case EnvironmentEnum.VMAS_SALP_NAVIGATE_LIDAR:
            env_config = SalpNavigateEnvironmentParams(**env_dict)

        case EnvironmentEnum.VMAS_SALP_PASSAGE:
            env_config = SalpPassageEnvironmentParams(**env_dict)

        case EnvironmentEnum.VMAS_BALANCE | EnvironmentEnum.VMAS_BUZZ_WIRE:
            env_config = EnvironmentParams(**env_dict)

        case (
            EnvironmentEnum.BOX2D_SALP
            | EnvironmentEnum.MPE_SPREAD
            | EnvironmentEnum.MPE_SIMPLE
        ):
            env_config = EnvironmentParams(**env_dict)

    env_config.environment = environment

    # Load experiment config
    exp_file = batch_dir / f"{experiment_name}.yaml"

    with open(exp_file, "r") as file:
        exp_dict = yaml.unsafe_load(file)

    match (algorithm):

        case AlgorithmEnum.PPO:
            exp_config = PPO_Experiment(**exp_dict)
            runner = PPO_Runner(
                exp_config.device,
                batch_dir,
                (Path(batch_dir).parents[1] / "results" / batch_name / experiment_name),
                trial_id,
                checkpoint,
                exp_config,
                env_config,
            )

    if view:
        runner.view()
    elif evaluate:
        runner.evaluate()
    else:
        runner.train()
