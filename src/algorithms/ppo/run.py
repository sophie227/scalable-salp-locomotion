from environments.types import EnvironmentParams
from algorithms.runner import Runner
from algorithms.ppo.types import Experiment
from algorithms.ppo.trainer import PPOTrainer
from algorithms.ppo.view import view
from algorithms.ppo.evaluate import evaluate

from pathlib import Path
import torch


class PPO_Runner(Runner):
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: str,
        checkpoint: bool,
        exp_config: Experiment,
        env_config: EnvironmentParams,
        curriculum: bool = False,
        curriculum_stage: int = None,
    ):
        super().__init__(device, batch_dir, trials_dir, trial_id, checkpoint)

        self.exp_config = exp_config
        self.env_config = env_config

        self.trainer = PPOTrainer(
            self.exp_config,
            self.env_config,
            self.device,
            self.trial_id,
            self.dirs,
            self.checkpoint,
            curriculum=curriculum,
            curriculum_stage=curriculum_stage,
        )

    def train(self):
        self.trainer.train()

    def view(self):
        view(self.exp_config, self.env_config, self.device, self.dirs)

    def evaluate(self):
        evaluate(
            self.exp_config,
            self.env_config,
            self.device,
            self.trial_id,
            self.dirs,
        )
