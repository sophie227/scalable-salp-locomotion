from vmas.simulator.dynamics.common import Dynamics
import torch


class SalpDynamics(Dynamics):
    @property
    def needed_action_size(self) -> int:
        return 2

    def process_action(self):
        pass
