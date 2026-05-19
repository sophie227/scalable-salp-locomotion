from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Optional

from algorithms.ppo.types import Experiment
from environments.types import EnvironmentParams
from algorithms.ppo.run import PPO_Runner


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
    stages: List[Dict],
    initial_checkpoint: Optional[Path] = None,
    view: bool = False,
    evaluate: bool = False,
):
    """
    Curriculum learning loop for PPO.

    Each stage:
    - clones base configs
    - applies environment patch
    - loads previous policy (if available)
    - continues training
    """

    last_checkpoint = initial_checkpoint

    print(f"[Curriculum] Starting training")
    print(f"[Curriculum] Initial checkpoint: {last_checkpoint}")

    for i, patch in enumerate(stages):
        stage_id = i + 1
        stage_trial_id = f"{trial_id}_stage_{stage_id}"

        print(f"\n=== Stage {stage_id}/{len(stages)}: {patch} ===")


        env_cfg = deepcopy(base_env)
        exp_cfg = deepcopy(base_exp)

        # Apply curriculum modification
        for k, v in patch.items():
            setattr(env_cfg, k, v)


        runner = PPO_Runner(
            device=device,
            batch_dir=batch_dir,
            trials_dir=trials_dir,
            trial_id=stage_trial_id,   
            checkpoint=False,           
            exp_config=exp_cfg,
            env_config=env_cfg,
        )

        # --------------------------------------------------
        # 3. Load previous policy (true curriculum transfer)
        # --------------------------------------------------
        if last_checkpoint is not None and last_checkpoint.exists():
            print(f"[Stage {stage_id}] Loading checkpoint: {last_checkpoint}")

            checkpoint = runner.trainer.learner.load(last_checkpoint)

        

        else:
            print(f"[Stage {stage_id}] No checkpoint loaded (training from scratch)")

    
        if view:
            runner.view()
        elif evaluate:
            runner.evaluate()
        else:
            runner.train()

 
        last_checkpoint = runner.trainer.dirs["models"] / "best_checkpoint.pt"

        if last_checkpoint.exists():
            print(f"[Stage {stage_id}] Saved best model: {last_checkpoint}")
        else:
            print(f"[Stage {stage_id}] WARNING: checkpoint not found!")

    return last_checkpoint