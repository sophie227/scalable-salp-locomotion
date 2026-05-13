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




from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from algorithms.ppo.run import PPO_Runner


# =========================================================
# 1. CURRICULUM STAGE DEFINITION
# =========================================================

@dataclass
class CurriculumStage:
    name: str
    env_overrides: Dict
    steps: int


# =========================================================
# 2. CHECKPOINT HELPERS (FULL RESTORE)
# =========================================================

def save_checkpoint(path: Path, model, optimizer, extra: dict = None):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path: Path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    return ckpt.get("extra", {})


# =========================================================
# 3. CURRICULUM RUNNER
# =========================================================

class CurriculumRunner:
    def __init__(
        self,
        base_exp,
        base_env,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: str,
        models_dir: Path,
    ):
        self.base_exp = base_exp
        self.base_env = base_env
        self.device = device

        self.batch_dir = batch_dir
        self.trials_dir = trials_dir
        self.trial_id = trial_id
        self.models_dir = models_dir

        self.last_checkpoint: Optional[Path] = None

    # -----------------------------------------------------
    # Run one stage
    # -----------------------------------------------------
    def run_stage(self, stage: CurriculumStage, stage_idx: int):

        print(f"\n==============================")
        print(f"STAGE {stage_idx}: {stage.name}")
        print(f"==============================")

        # ---- clone configs ----
        env_cfg = deepcopy(self.base_env)
        exp_cfg = deepcopy(self.base_exp)

        # ---- apply overrides ----
        for k, v in stage.env_overrides.items():
            setattr(env_cfg, k, v)

        # ---- build runner ----
        runner = PPO_Runner(
            device=self.device,
            batch_dir=self.batch_dir,
            trials_dir=self.trials_dir,
            trial_id=f"{self.trial_id}_stage_{stage_idx}",
            checkpoint=True,
            exp_config=exp_cfg,
            env_config=env_cfg,
            curriculum=True,
            curriculum_stage=stage_idx,
            initial_checkpoint=self.last_checkpoint,   
        )

        trainer = runner.trainer

        # ---- train ----
        trainer.train()

        # ---- save checkpoint ----
        stage_ckpt = self.models_dir / f"stage_{stage_idx}.pt"

        save_checkpoint(
            stage_ckpt,
            trainer.learner.policy,
            trainer.learner.optimizer,
            extra={
                "stage": stage.name,
                "env": stage.env_overrides,
            },
        )

        print(f"Saved stage checkpoint → {stage_ckpt}")

        self.last_checkpoint = stage_ckpt

        return stage_ckpt

    # -----------------------------------------------------
    # Run full curriculum
    # -----------------------------------------------------
    def run(self, curriculum: List[CurriculumStage]):
        for i, stage in enumerate(curriculum):
            self.run_stage(stage, i + 1)


# =========================================================
# 4. EXAMPLE CURRICULUM (3 LEVELS)
# =========================================================

def build_default_curriculum():
    return [
        CurriculumStage(
            name="easy",
            env_overrides={
                "neighbor_offset": "large",
            },
            steps=2_000_000,
        ),
        CurriculumStage(
            name="medium",
            env_overrides={
                "neighbor_offset": "medium",
            },
            steps=1_000_000,
        ),

        CurriculumStage(
            name="hard",
            env_overrides={
                "neighbor_offset": "small",
            },
            steps=1_000_000,
        ),
        CurriculumStage(
            name="expert",
            env_overrides={
                "neighbor_offset": "expert",
            },
            steps=2_000_000,
        ),
    ]


# =========================================================
# 5. ENTRY POINT
# =========================================================

def run_curriculum_training(
    base_exp,
    base_env,
    device,
    batch_dir,
    trials_dir,
    trial_id,
    models_dir,
):
    runner = CurriculumRunner(
        base_exp=base_exp,
        base_env=base_env,
        device=device,
        batch_dir=batch_dir,
        trials_dir=trials_dir,
        trial_id=trial_id,
        models_dir=models_dir,
    )

    curriculum = build_default_curriculum()
    runner.run(curriculum)

    return runner.last_checkpoint