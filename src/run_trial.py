from algorithms.algorithms import run_algorithm
from algorithms.ppo.cirriculum import run_curriculum
import yaml
from environments.types import EnvironmentEnum
from environments.salp_navigate.types import SalpNavigateEnvironmentParams
from algorithms.ppo.types import Experiment as PPO_Experiment


import argparse
from pathlib import Path

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
        "--algorithm",
        default="",
        help="Learning algorithm name",
        type=str,
    )
    parser.add_argument(
        "--environment",
        default="",
        help="Learning environment name",
        type=str,
    )

    parser.add_argument(
        "--view",
        action="store_true",
        help="Runs view method instead of train",
    )

    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="Load model checkpointfor training",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run validation script",
    )

    parser.add_argument("--trial_id", default="1", help="Sets trial ID", type=str)

    # curriculum options
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum learning (multiple stages)",
    )
    parser.add_argument(
        "--stages",
        default=None,
        help="YAML/JSON string describing list of stage patches, e.g. '[{\"n_agents\":8},{\"n_agents\":12}]'",
        type=str,
    )
    # parser.add_argument(
    #     "--initial_checkpoint",
    #     default=None,
    #     help="Path to a pretrained model (best_model) to start curriculum from",
    #     type=str,
    # )
    # allow viewing/evaluating the curriculum rather than training
    parser.add_argument(
        "--curriculum_view",
        action="store_true",
        help="Run curriculum stages in view mode instead of training",
    )
    parser.add_argument(
        "--curriculum_evaluate",
        action="store_true",
        help="Run curriculum stages in evaluate mode instead of training",
    )

    args = vars(parser.parse_args())

    # Set base_config path
    dir_path = Path(__file__).parent

    # Set configuration folder
    batch_dir = dir_path / "experiments" / "yamls" / args["batch"]

    # load environment and experiment configurations so we can pass them
    
    env_file = batch_dir / "_env.yaml"
    with open(env_file, "r") as file:
        env_dict = yaml.safe_load(file)

    # dispatch based on environment string; this mirrors run_algorithm
    match args["environment"]:
        case EnvironmentEnum.VMAS_SALP_NAVIGATE:
            base_env = SalpNavigateEnvironmentParams(**env_dict)
        case EnvironmentEnum.VMAS_SALP_NAVIGATE_LIDAR:
            base_env = SalpNavigateEnvironmentParams(**env_dict)
        case EnvironmentEnum.VMAS_SALP_PASSAGE:
            base_env = SalpNavigateEnvironmentParams(**env_dict)
        case _:
            # use generic params for other cases
            from environments.types import EnvironmentParams

            base_env = EnvironmentParams(**env_dict)
    base_env.environment = args["environment"]

    exp_file = batch_dir / f"{args['name']}.yaml"
    with open(exp_file, "r") as file:
        exp_dict = yaml.safe_load(file)
    base_exp = PPO_Experiment(**exp_dict)

    trials_dir = Path(batch_dir).parents[1] / "results" / args["batch"] / args["name"]

    # decide whether to run curriculum or standard training
    if args.get("curriculum"):
        # stages may be provided as a YAML string or loaded from a file
        stages = []
        if args.get("stages"):
            stages = yaml.safe_load(args.get("stages"))

        initial_ckpt = None
        if args.get("initial_checkpoint"):
            initial_ckpt = Path(args.get("initial_checkpoint"))
        else:
            # default to latest checkpoint from previous standard run (if any)
            default_ckpt_path = (
                trials_dir / args["trial_id"] / "models" / "checkpoint"
            )
            if default_ckpt_path.is_file():
                initial_ckpt = default_ckpt_path

        run_curriculum(
            base_env=base_env,
            base_exp=base_exp,
            device=base_exp.device,
            batch_dir=batch_dir,
            batch_name=args["batch"],
            experiment_name=args["name"],
            environment=args["environment"],
            algorithm=args["algorithm"],
            trials_dir=trials_dir,
            trial_id=args["trial_id"],
            stages=stages,
            initial_checkpoint=initial_ckpt,
            view=args.get("curriculum_view"),
            evaluate=args.get("curriculum_evaluate"),
        )
    else:
        # Run learning algorithm normally
        run_algorithm(
            batch_dir=batch_dir,
            batch_name=args["batch"],
            experiment_name=args["name"],
            trial_id=args["trial_id"],
            algorithm=args["algorithm"],
            environment=args["environment"],
            view=args["view"],
            checkpoint=args["checkpoint"],
            evaluate=args["evaluate"],
        )
