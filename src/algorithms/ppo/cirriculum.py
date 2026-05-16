from copy import deepcopy
from pathlib import Path
from typing import List, Dict, Optional
from torch.optim import Adam

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
    initial_checkpoint: Optional[Path] = None, #'/home/sophie/scalable-salp-locomotion/src/experiments/results/salp_navigate_5a/gcn/0/models/best_model',
    view: bool = False,
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

    success_thresh = .90
    max_attempts = 20

    for i, patch in enumerate(stages):
        print(f"\ncurriculum stage {i+1}/{len(stages)}: {patch}")
        env_cfg = deepcopy(base_env)
        exp_cfg = deepcopy(base_exp)

        for k, v in patch.items():
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
        )

        if last_checkpoint is not None and last_checkpoint.is_file():
            runner.trainer.learner.load(last_checkpoint)

        stage_complete = False
        attempt = 0
        
        while not stage_complete:
            attempt += 1
            runner.trainer.train()
            success_rate = runner.evaluate()
            print(f" Stage {i+1} Success rate: {success_rate:.2%}")

            stage_checkpoint = runner.trainer.dirs["models"] / f"stage_{i+1}_checkpoint.pt"
            last_checkpoint = stage_checkpoint

            if success_rate >= success_thresh:
                print(f"Stage {i+1} complete with success rate {success_rate:.2%}")
                stage_complete = True
            
            elif attempt >= max_attempts:
                print(f"Stage {i+1} failed to reach success threshold after {attempt} attempts. Moving to next stage.")
                stage_complete = True

            # # IMPORTANT: reset optimizer
            # runner.trainer.learner.optimizer = Adam(
            #     runner.trainer.learner.policy.parameters(),
            #     lr=exp_cfg.params["lr"]
            # )


    return last_checkpoint



# from copy import deepcopy
# from pathlib import Path
# from typing import List, Dict, Optional

# from algorithms.ppo.types import Experiment
# from environments.types import EnvironmentParams
# from algorithms.ppo.run import PPO_Runner


# def run_curriculum(
#     base_exp: Experiment,
#     base_env: EnvironmentParams,
#     device: str,
#     batch_dir: Path,
#     batch_name: str,
#     experiment_name: str,
#     environment: str,
#     algorithm: str,
#     trials_dir: Path,
#     trial_id: str,
#     stages: List[Dict],
#     initial_checkpoint: Optional[Path] = None,
#     curriculum_view: bool = False,
#     evaluate: bool = False,
# ):
#     last_checkpoint = initial_checkpoint

#     print("start")
#     print(f"initial checkpoint: {last_checkpoint}")

#     for i, patch in enumerate(stages):
#         print(f"\n=== curriculum stage {i+1}/{len(stages)}: {patch} ===")

#         if i == 0:
#             env_cfg = deepcopy(base_env)
#         else:
#             env_cfg = deepcopy(prev_env_cfg)

#         exp_cfg = deepcopy(base_exp)
        
#         for k, v in patch.items():
#             setattr(env_cfg, k, v)

#         stage_trial_id = f"{trial_id}_stage_{i+1}"

#         runner = PPO_Runner(
#             device=device,
#             batch_dir=batch_dir,
#             trials_dir=trials_dir,
#             trial_id=stage_trial_id,
#             checkpoint=True,
#             exp_config=exp_cfg,
#             env_config=env_cfg,
#             curriculum=True,
#             curriculum_stage=i + 1,
#         )

#         # -----------------------------
#         # LOAD PREVIOUS STAGE CHECKPOINT
#         # -----------------------------
#         if last_checkpoint is not None and Path(last_checkpoint).is_file():
#             print(f"Loading checkpoint: {last_checkpoint}")
#             runner.trainer.learner.load(last_checkpoint)

#             print(
#                 "Loaded weight:",
#                 next(
#                     runner.trainer.learner.policy.parameters()
#                 ).flatten()[0].item()
#             )
#         # -----------------------------
#         # RUN STAGE
#         # -----------------------------
#         if curriculum_view:
#             runner.view()
#         elif evaluate:
#             runner.evaluate()
#         else:
#             runner.train()

#         # -----------------------------
#         # SAVE STAGE CHECKPOINT
#         # -----------------------------
#         stage_checkpoint = runner.trainer.dirs["models"] / "checkpoint.pt"

#         print(f"Finished stage {i+1}")
#         print(f"Saved: {stage_checkpoint}")

#         # IMPORTANT: pass forward correctly
#         last_checkpoint = stage_checkpoint
#         prev_env_cfg = deepcopy(env_cfg)

#     return last_checkpoint



# from copy import deepcopy
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List, Optional

# import torch

# from algorithms.ppo.types import Experiment
# from environments.types import EnvironmentParams
# from algorithms.ppo.run import PPO_Runner


# # =========================================================
# # 1. CURRICULUM STAGE
# # =========================================================

# @dataclass
# class CurriculumStage:
#     name: str
#     env_overrides: Dict
#     steps: int


# # =========================================================
# # 2. CHECKPOINT HELPERS
# # =========================================================

# def save_checkpoint(path: Path, learner, extra: dict = None):

#     checkpoint = {
#         "model": learner.policy.state_dict(),
#         "optimizer": learner.optimizer.state_dict(),
#         "extra": extra or {},
#     }

#     torch.save(checkpoint, path)


# def load_checkpoint(path: Path, learner):

#     checkpoint = torch.load(path, map_location="cpu")

#     learner.policy.load_state_dict(checkpoint["model"])
#     learner.optimizer.load_state_dict(checkpoint["optimizer"])

#     return checkpoint.get("extra", {})


# # =========================================================
# # 3. CURRICULUM RUNNER
# # =========================================================

# def run_curriculum(
#     base_exp: Experiment,
#     base_env: EnvironmentParams,
#     device: str,
#     batch_dir: Path,
#     batch_name: str,
#     experiment_name: str,
#     environment: str,
#     algorithm: str,
#     trials_dir: Path,
#     trial_id: str,
#     stages: List[Dict],
#     initial_checkpoint: Optional[Path] = None,
#     curriculum_view: bool = False,
#     evaluate: bool = False,
# ):

#     print("====================================")
#     print("STARTING CURRICULUM TRAINING")
#     print("====================================")

#     # -----------------------------------------------------
#     # Build curriculum stages
#     # -----------------------------------------------------

#     curriculum_stages: List[CurriculumStage] = []

#     for i, stage in enumerate(stages):

#         curriculum_stages.append(
#             CurriculumStage(
#                 name=f"stage_{i+1}",
#                 env_overrides=stage,
#                 steps=base_exp.params.n_total_steps,
#             )
#         )

#     # -----------------------------------------------------
#     # Initial checkpoint
#     # -----------------------------------------------------

#     last_checkpoint = initial_checkpoint

#     print(f"Initial checkpoint: {last_checkpoint}")

#     prev_env_cfg = deepcopy(base_env)

#     # =====================================================
#     # CURRICULUM LOOP
#     # =====================================================

#     for stage_idx, stage in enumerate(curriculum_stages, start=1):

#         print("\n====================================")
#         print(f"CURRICULUM STAGE {stage_idx}")
#         print(f"Stage name: {stage.name}")
#         print(f"Overrides: {stage.env_overrides}")
#         print("====================================")

#         # -------------------------------------------------
#         # Clone configs
#         # -------------------------------------------------

#         env_cfg = deepcopy(prev_env_cfg)
#         exp_cfg = deepcopy(base_exp)

#         # -------------------------------------------------
#         # Apply stage overrides
#         # -------------------------------------------------

#         for k, v in stage.env_overrides.items():
#             setattr(env_cfg, k, v)

#         # -------------------------------------------------
#         # Curriculum scheduling
#         # -------------------------------------------------

#         exp_cfg.params.n_total_steps = stage.steps

#         # Lower LR as stages get harder
#         exp_cfg.params.lr *= (0.5 ** (stage_idx - 1))

#         # Lower entropy as policy stabilizes
#         exp_cfg.params.ent_coef *= (0.7 ** (stage_idx - 1))

#         print(f"LR: {exp_cfg.params.lr}")
#         print(f"Entropy coef: {exp_cfg.params.ent_coef}")
#         print(f"Steps: {exp_cfg.params.n_total_steps}")

#         # -------------------------------------------------
#         # Stage-specific trial ID
#         # -------------------------------------------------

#         stage_trial_id = f"{trial_id}_stage_{stage_idx}"

#         # -------------------------------------------------
#         # Create PPO runner
#         # -------------------------------------------------

#         runner = PPO_Runner(
#             device=device,
#             batch_dir=batch_dir,
#             trials_dir=trials_dir,
#             trial_id=stage_trial_id,
#             checkpoint=True,
#             exp_config=exp_cfg,
#             env_config=env_cfg,
#             curriculum=True,
#             curriculum_stage=stage_idx,
#         )

#         trainer = runner.trainer

#         # -------------------------------------------------
#         # LOAD PREVIOUS STAGE
#         # -------------------------------------------------

#         if last_checkpoint is not None and Path(last_checkpoint).exists():

#             print("\n------------------------------------")
#             print("LOADING PREVIOUS STAGE CHECKPOINT")
#             print("------------------------------------")

#             print(f"Checkpoint: {last_checkpoint}")

#             weight_before = next(
#                 trainer.learner.policy.parameters()
#             ).flatten()[0].item()

#             print(f"Weight BEFORE load: {weight_before}")

#             try:

#                 load_checkpoint(
#                     last_checkpoint,
#                     trainer.learner,
#                 )

#                 weight_after = next(
#                     trainer.learner.policy.parameters()
#                 ).flatten()[0].item()

#                 print(f"Weight AFTER load: {weight_after}")

#                 if weight_before == weight_after:
#                     print("WARNING: weights did NOT change")
#                 else:
#                     print("SUCCESS: weights restored")

#             except Exception as exc:

#                 print("FAILED TO LOAD CHECKPOINT")
#                 print(exc)

#             print("------------------------------------\n")

#         else:

#             print("No previous checkpoint found")

#         # -------------------------------------------------
#         # RUN STAGE
#         # -------------------------------------------------

#         if curriculum_view:

#             print("VIEWING CURRICULUM STAGE")
#             runner.view()

#         elif evaluate:

#             print("EVALUATING CURRICULUM STAGE")
#             runner.evaluate()

#         else:

#             print("TRAINING CURRICULUM STAGE")
#             trainer.train()

#         # -------------------------------------------------
#         # SAVE CURRENT STAGE
#         # -------------------------------------------------

#         stage_model_dir = trainer.dirs["models"]

#         stage_checkpoint = (
#             stage_model_dir / f"curriculum_stage_{stage_idx}.pt"
#         )

#         print("\n------------------------------------")
#         print("SAVING CURRICULUM CHECKPOINT")
#         print("------------------------------------")

#         save_checkpoint(
#             stage_checkpoint,
#             trainer.learner,
#             extra={
#                 "stage_idx": stage_idx,
#                 "stage_name": stage.name,
#                 "env_overrides": stage.env_overrides,
#             },
#         )

#         print(f"Saved checkpoint: {stage_checkpoint}")

#         # -------------------------------------------------
#         # Pass checkpoint forward
#         # -------------------------------------------------

#         last_checkpoint = stage_checkpoint

#         # -------------------------------------------------
#         # Carry forward env config
#         # -------------------------------------------------

#         prev_env_cfg = deepcopy(env_cfg)

#         print("------------------------------------")

#     # =====================================================
#     # DONE
#     # =====================================================

#     print("\n====================================")
#     print("CURRICULUM TRAINING FINISHED")
#     print("====================================")

#     print(f"Final checkpoint: {last_checkpoint}")

#     return last_checkpoint