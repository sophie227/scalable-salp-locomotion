
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Optional

from algorithms import runner
from algorithms.ppo.types import Experiment
from environments.types import EnvironmentParams
from algorithms.ppo.run import PPO_Runner        # or import PPOTrainer directly

def run_curriculum(
    base_exp: Experiment,
    base_env: EnvironmentParams,
    device: str,
    batch_dir: Path,
    batch_name: str,
    experiment_name: str,
    environment: str,
    algorithm: str,
    trials_dir: Path,
    trial_id: str,
    stages: List[Dict],              # e.g. [{'n_agents':8}, {'n_agents':12}, {'n_agents':16}]
    # initial_checkpoint: Optional[Path] = Path("/experiments/results/salp_navigate_5a_1t/gcn/0/models/checkpoint"), #'/home/sophie/scalable-salp-locomotion/src/experiments/results/salp_navigate_5a/gcn/0/models/checkpoint',
    initial_checkpoint: Optional[Path] = None,
    curriculum_view: bool = False,
    evaluate: bool = False,
):



    """
    Run a sequence of training stages.  At each stage the environment
    configuration is patched with the dict in `stages`, the trainer is
    constructed, and – if we have a previous checkpoint – it is loaded.

    The returned Path points to the last best‑model file.
    """
    last_checkpoint = initial_checkpoint
    print("start")
    print(f"initial checkpoint: {last_checkpoint}")

    for i, patch in enumerate(stages):
        print(f"\n=== curriculum stage {i+1}/{len(stages)}: {patch} ===")

        # clone configs so we don’t mutate the caller’s copy
        env_cfg = deepcopy(base_env)
        exp_cfg = deepcopy(base_exp)

        # apply the stage-specific overrides
        for k, v in patch.items():
            if k == "chain":
                setattr(env_cfg, "n_agents", v)
            else:
                setattr(env_cfg, k, v)

        stage_trial_id = f"{trial_id}_stage_{i+1}"

        runner = PPO_Runner(
            device=device,
            batch_dir=batch_dir,
            trials_dir=trials_dir,
            trial_id=stage_trial_id,
            checkpoint=False,
            exp_config=exp_cfg,
            env_config=env_cfg,
            curriculum=True,
            curriculum_stage=i+1,
        )

        # if we’ve trained something already, load it
        # if we’ve trained something already, load it

        # if we’ve trained something already, load it
        if last_checkpoint is not None and last_checkpoint.is_file():

            print("\n====================")
            print("LOADING CHECKPOINT")
            print("====================")

            print(f"Checkpoint path: {last_checkpoint}")
            print(f"Checkpoint exists: {last_checkpoint.exists()}")

            # Weight BEFORE loading
            first_weight_before = next(
                runner.trainer.learner.policy.parameters()
            ).flatten()[0].item()

            print(f"Weight BEFORE load: {first_weight_before}")

            try:
                runner.trainer.learner.load(last_checkpoint)

                # Weight AFTER loading
                first_weight_after = next(
                    runner.trainer.learner.policy.parameters()
                ).flatten()[0].item()

                print(f"Weight AFTER load: {first_weight_after}")

                if first_weight_before == first_weight_after:
                    print("WARNING: weights did NOT change after loading")
                else:
                    print("SUCCESS: weights changed after loading")

                print(f"Loaded checkpoint: {last_checkpoint}")

            except RuntimeError as exc:
                print(
                    "Skipping checkpoint load due to shape mismatch:\n"
                    f"{exc}"
                )

            print("====================\n")


        # run one full experiment (honours exp_cfg.params.n_total_steps, etc)
        if curriculum_view:

            current_stage_checkpoint = (
                runner.trainer.dirs["models"] / "best_checkpoint.pt"
            )

            if current_stage_checkpoint.exists():

                print(f"Loading current stage checkpoint: {current_stage_checkpoint}")

                runner.trainer.learner.load(current_stage_checkpoint)

            else:
                print(f"No checkpoint found for stage {i+1}")

            runner.view()

        elif evaluate:

            current_stage_checkpoint = (
                runner.trainer.dirs["models"] / "best_checkpoint.pt"
            )

            if current_stage_checkpoint.exists():

                print(f"Loading current stage checkpoint: {current_stage_checkpoint}")

                runner.trainer.learner.load(current_stage_checkpoint)

            else:
                print(f"No checkpoint found for stage {i+1}")

            runner.evaluate()

        else:
            runner.train()
        # after training save file will be at
       # Path to this stage's model directory
        stage_model_dir = runner.trainer.dirs["models"]

        # This is the BEST model of the CURRENT stage
        stage_best_checkpoint = stage_model_dir / "best_checkpoint.pt"

        print(f"Finished stage {i+1}")
        print(f"Saving CURRENT stage best model: {stage_best_checkpoint}")

        # This will be used for the NEXT stage
        last_checkpoint = stage_best_checkpoint
        print(f"Checkpoint for next stage: {last_checkpoint}")

    return last_checkpoint