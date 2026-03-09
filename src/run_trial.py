from algorithms.algorithms import run_algorithm

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

    args = vars(parser.parse_args())

    # Set base_config path
    dir_path = Path(__file__).parent

    # Set configuration folder
    batch_dir = dir_path / "experiments" / "yamls" / args["batch"]

    # Run learning algorithm
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
