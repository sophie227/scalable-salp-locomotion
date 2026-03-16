
from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Optional

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
    initial_checkpoint: Optional[Path] = None,
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
            setattr(env_cfg, k, v)

        runner = PPO_Runner(
            device=device,
            batch_dir=batch_dir,
            trials_dir=trials_dir,
            trial_id=trial_id,
            checkpoint=False,           # we handle loading ourselves
            exp_config=exp_cfg,
            env_config=env_cfg,
        )

        # if we’ve trained something already, load it
        if last_checkpoint is not None and last_checkpoint.is_file():
            runner.trainer.learner.load(last_checkpoint)


        # run one full experiment (honours exp_cfg.params.n_total_steps, etc)
        if view:
            runner.view()
        elif evaluate:
            runner.evaluate()
        else:
            runner.train()

        # after training save file will be at
        last_checkpoint = runner.trainer.dirs["models"] / "best_cirr_model"
        print(f"Finished stage {i+1}, best model saved to {last_checkpoint}")

    return last_checkpoint