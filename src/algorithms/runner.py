from pathlib import Path
import torch


class Runner:
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: str,
        checkpoint: bool,
    ):
        # Directories
        self.device = device
        self.trial_id = trial_id
        self.batch_dir = batch_dir
        self.trial_dir = trials_dir / trial_id
        self.logs_dir = self.trial_dir / "logs"
        self.models_dir = self.trial_dir / "models"
        self.video_dir = self.trial_dir / "videos"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        # Create directories dictionary
        self.dirs = {
            "batch": batch_dir,
            "logs": self.logs_dir,
            "models": self.models_dir,
            "videos": self.video_dir,
        }

        # Checkpoint loading
        self.checkpoint = checkpoint

        # Set thread settings
        n_threads = 1
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(n_threads)
        print(f"PyTorch using {torch.get_num_threads()} threads")

    def train(self):
        pass

    def view(self):
        pass

    def evaluate(self):
        pass
