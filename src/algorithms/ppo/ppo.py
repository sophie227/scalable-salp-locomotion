import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from algorithms.ppo.types import Params

import dill

# Useful for error tracing
torch.autograd.set_detect_anomaly(True)


class TensorRolloutBuffer:
    def __init__(self, model, batch_size, n_envs, n_agents, d_state, d_action, device):
        # Calculate buffer size
        steps = batch_size // n_envs

        # Pre-allocate tensors
        if model == "mlp":
            self.states = torch.zeros((steps, n_envs, d_state), device=device)
        else:
            self.states = torch.zeros((steps, n_envs, n_agents, d_state), device=device)

        self.actions = torch.zeros((steps, n_envs, n_agents * d_action), device=device)
        self.logprobs = torch.zeros((steps, n_envs, n_agents), device=device)
        self.rewards = torch.zeros((steps, n_envs), device=device)
        self.values = torch.zeros((steps, n_envs, 1), device=device)
        self.is_terminals = torch.zeros(
            (steps, n_envs), dtype=torch.bool, device=device
        )

        self.step = 0  # Current position in buffer
        self.capacity = steps

    def add(self, state, action, logprob, value, reward, done):
        if self.step >= self.capacity:
            return  

        self.states[self.step] = state
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.values[self.step] = value
        self.rewards[self.step] = reward
        self.is_terminals[self.step] = done

        self.step += 1

    def clear(self):
        self.step = 0


class RolloutData(Dataset):
    def __init__(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        values: torch.Tensor,
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
        rewards: torch.Tensor,
    ):
        self.states = states
        self.actions = actions
        self.values = values
        self.logprobs = logprobs
        self.advantages = advantages
        self.rewards = rewards

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.values[idx],
            self.logprobs[idx],
            self.advantages[idx],
            self.rewards[idx],
        )


class PPO:
    def __init__(
        self,
        device: str,
        model: str,
        params: Params,
        n_agents_train: int,
        n_agents_eval: int,
        n_envs: int,
        d_state: int,
        d_action: int,
        checkpoint: bool = False,
    ):
        self.device = device
        self.checkpoint = checkpoint
        self.n_envs = n_envs
        self.n_agents = n_agents_train
        self.d_action = d_action
        self.buffer = TensorRolloutBuffer(
            model, params.batch_size, n_envs, n_agents_train, d_state, d_action, device
        )

        # Algorithm parameters
        self.n_epochs = params.n_epochs
        self.minibatch_size = params.batch_size // params.n_minibatches
        self.gamma = params.gamma
        self.lmbda = params.lmbda
        self.eps_clip = params.eps_clip
        self.grad_clip = params.grad_clip
        self.ent_coef = params.ent_coef
        self.std_coef = params.std_coef
        self.n_epochs = params.n_epochs
        self.use_clipped_value_loss = True

        # Select model
        match (model):
            case "mlp":
                from algorithms.ppo.models.mlp_ac import ActorCritic
            case "transformer":
                from algorithms.ppo.models.transformer_ac import ActorCritic
            case "transformer_full":
                from algorithms.ppo.models.transformer_full_ac import (
                    ActorCritic,
                )
            case "transformer_encoder":
                from algorithms.ppo.models.transformer_encoder_ac import (
                    ActorCritic,
                )
            case "transformer_decoder":
                from algorithms.ppo.models.transformer_decoder_ac import (
                    ActorCritic,
                )
            case "gcn":
                from algorithms.ppo.models.gcn_ac import ActorCritic

            case "gcn_dim":
                from algorithms.ppo.models.gcn_ac_dim import ActorCritic
            case "gat":
                from algorithms.ppo.models.gat_ac import ActorCritic
            case "graph_transformer":
                from algorithms.ppo.models.graph_transformer_ac import (
                    ActorCritic,
                )
            case "gcn_full":
                from algorithms.ppo.models.gcn_full_ac import ActorCritic
            case "gat_full":
                from algorithms.ppo.models.gat_full_ac import ActorCritic

            case "graph_transformer_full":
                from algorithms.ppo.models.graph_transformer_full_ac import (
                    ActorCritic,
                )

        # Create models
        self.policy = ActorCritic(
            n_agents_train,
            n_agents_eval,
            d_state,
            d_action,
            self.device,
        ).to(self.device)

        self.policy_old = ActorCritic(
            n_agents_train,
            n_agents_eval,
            d_state,
            d_action,
            self.device,
        ).to(self.device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        # Create optimizers
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=params.lr,
        )

        # Logging params
        self.total_epochs = 0

    def calc_value_loss(self, values, value_preds_batch, return_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """

        clip_param = 0.05
        huber_delta = 10

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -clip_param, clip_param
        )

        value_loss_clipped = F.huber_loss(
            return_batch, value_pred_clipped, delta=huber_delta
        )
        value_loss_original = F.huber_loss(return_batch, values, delta=huber_delta)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss

    def select_action(self, state):
        with torch.no_grad():
            action, action_logprob, state_val = self.policy_old.act(state)
        return state, action, action_logprob, state_val

    def deterministic_action(self, state):
        with torch.no_grad():
            action = self.policy_old.act(state, deterministic=True)

        return action.detach()

    def gae(self):
        # Get next value for bootstrapping
        with torch.no_grad():
            next_value = (
                self.policy_old.get_value(self.buffer.states[-1])
                * (~self.buffer.is_terminals[-1]).unsqueeze(-1).float()
            )

        # Prepare data for vectorized computation
        values = torch.cat([self.buffer.values, next_value.unsqueeze(0)]).squeeze(-1)
        mask = (~self.buffer.is_terminals).float()

        # Vectorized GAE calculation
        advantages = torch.zeros_like(self.buffer.rewards)
        gae = torch.zeros(self.n_envs, device=self.device)

        for t in reversed(range(self.buffer.step)):
            delta = (
                self.buffer.rewards[t]
                + self.gamma * values[t + 1] * mask[t]
                - values[t]
            )
            gae = delta + self.gamma * self.lmbda * mask[t] * gae
            advantages[t] = gae

        returns = advantages + values[:-1, :]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Reshape for training (no need for transposing with tensor buffer)
        return (advantages.reshape(-1, 1).detach(), returns.reshape(-1, 1).detach())

    def update(self):

        # Get advantages and returns
        advantages, returns = self.gae()

        # Reshape data directly - no need for stacking operations
        old_states = self.buffer.states.flatten(0, 1)
        old_actions = self.buffer.actions.flatten(0, 1)
        old_values = self.buffer.values.flatten(0, 1)
        old_logprobs = self.buffer.logprobs.flatten(0, 1)

        # Create dataset from rollout
        dataset = RolloutData(
            old_states, old_actions, old_values, old_logprobs, advantages, returns
        )

        loader = DataLoader(
            dataset,
            batch_size=self.minibatch_size,
            shuffle=True,
        )

        # Load model into GPU for training
        train_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.policy.to(train_device)
            self.policy.device = train_device

        # Optimize policy for n epochs
        for _ in range(self.n_epochs):

            for (
                b_old_states,
                b_old_actions,
                b_old_values,
                b_old_logprobs,
                b_advantages,
                b_returns,
            ) in loader:

                # Load batch into GPU for training
                if torch.cuda.is_available():
                    b_old_states = b_old_states.to(train_device)
                    b_old_actions = b_old_actions.to(train_device)
                    b_old_values = b_old_values.to(train_device)
                    b_old_logprobs = b_old_logprobs.to(train_device)
                    b_advantages = b_advantages.to(train_device)
                    b_returns = b_returns.to(train_device)

                # Evaluating old actions and values
                logprobs, values, dist_entropy = self.policy.evaluate(
                    b_old_states, b_old_actions
                )

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - b_old_logprobs)

                # Finding Surrogate Loss
                surr1 = ratios * b_advantages
                surr2 = (
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip)
                    * b_advantages
                )

                # Penalize high values of log_std by increasing the loss, thus decreasing exploration
                log_std_penalty = (
                    self.std_coef
                    * self.policy.log_action_std[: self.n_agents * self.d_action]
                    .square()
                    .mean()
                )

                # Promote exploration by reducing the loss if entropy increases
                entropy_bonus = self.ent_coef * dist_entropy.mean()

                ppo_loss = -torch.min(surr1, surr2).mean()

                # Calculate actor and critic losses
                actor_loss = ppo_loss + log_std_penalty - entropy_bonus
                value_loss = self.calc_value_loss(values, b_old_values, b_returns)

                loss = actor_loss + value_loss

                # Take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                self.optimizer.step()

                # Store data
                self.total_epochs += 1

        # Load model back to cpu to collect rollouts
        self.policy.to(self.device)
        self.policy.device = self.device

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    # def load(self, checkpoint_path):
        # self.policy_old.load_state_dict(
        #     torch.load(
        #         checkpoint_path,
        #         map_location=lambda storage, loc: storage,
        #         weights_only=True,
        #     )
        # )
        # self.policy.load_state_dict(
        #     torch.load(
        #         checkpoint_path,
        #         map_location=lambda storage, loc: storage,
        #         weights_only=True,
        #     )
        # )

    def load(self, checkpoint_path):
        checkpoint_state = torch.load(
            checkpoint_path,
            map_location=lambda storage, loc: storage,
            weights_only=True,
        )

        current_state = self.policy.state_dict()
        compatible_state = {}
        skipped = []

        for key, value in checkpoint_state.items():
            if key not in current_state:
                skipped.append(key)
                continue

            target = current_state[key]

            if target.shape == value.shape:
                compatible_state[key] = value
                continue

            if target.ndim == value.ndim:
                resized = target.clone()
                overlap = tuple(slice(0, min(a, b)) for a, b in zip(target.shape, value.shape))
                resized[overlap] = value[overlap]
                compatible_state[key] = resized
                continue

            skipped.append(key)

        self.policy.load_state_dict(compatible_state, strict=False)
        self.policy_old.load_state_dict(self.policy.state_dict())

        if skipped:
            print(
                "Partial checkpoint load: skipped incompatible parameters "
                f"({len(skipped)}), e.g. {skipped[:5]}"
            )
    
        