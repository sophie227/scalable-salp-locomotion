import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torch_geometric.nn import AttentionalAggregation

from algorithms.ppo.models.utils import create_graph_batch


class ActorCritic(torch.nn.Module):
    def __init__(
        self,
        n_agents_train: int,
        n_agents_eval: int,
        d_state: int,
        d_action: int,
        device: str,
        hidden_dim=128,
        graph_type="chain",
    ):
        super(ActorCritic, self).__init__()

        self.n_agents_eval = n_agents_eval
        self.d_action = d_action
        self.d_state = d_state
        self.device = device
        self.graph_type = graph_type

        # self.log_action_std = nn.Parameter(
        #     torch.ones(
        #         d_action * n_agents_train,
        #         requires_grad=True,
        #         device=device,
        #     )
        #     * -0.5
        # )
        self.log_action_std = nn.Parameter(
         torch.ones(d_action, device=device) * -0.5
)

        # GCN layers instead of GAT
        self.gcn1 = GCNConv(d_state, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, d_action),
        )

        # Critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.att_pool = AttentionalAggregation(
            nn.Sequential(nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, 1))
        )

    def get_action_and_value(self, state):

        graph_list = create_graph_batch(state, self.graph_type, self.device)

        batched_graph = Batch.from_data_list(graph_list)

        x = self.forward(batched_graph)

        graph_emb = self.att_pool(x, batched_graph.batch)
        value = self.value_head(graph_emb)

        action_mean = (
            self.actor_head(x)
            .reshape((state.shape[0], state.shape[1], self.d_action))
            .flatten(start_dim=1)
        )

        return action_mean, value

    def forward(self, batch: Batch):
        x = self.gcn1(batch.x, batch.edge_index)
        x = F.gelu(x)

        # Normalization is important for GCN training stability
        x = F.layer_norm(x, x.shape[1:])

        x = self.gcn2(x, batch.edge_index)
        x = F.gelu(x)

        return x

    def get_value(self, state: torch.Tensor):
        with torch.no_grad():
            _, value = self.get_action_and_value(state)
            return value

    # def get_action_dist(self, action_mean):
    #     action_std = torch.exp(self.log_action_std[: action_mean.shape[-1]])
    #     return Normal(action_mean, action_std)

    def get_action_dist(self, action_mean):
        action_std = torch.exp(self.log_action_std).unsqueeze(0)
        action_std = action_std.expand_as(action_mean)
        return Normal(action_mean, action_std)

    def act(self, state, deterministic=False):

        action_mean, value = self.get_action_and_value(state)

        if deterministic:
            return action_mean.detach()

        dist = self.get_action_dist(action_mean)

        action = dist.sample()
        action_logprob = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)

        return (
            action.detach(),
            action_logprob.detach(),
            value.detach(),
        )

    def evaluate(self, state, action):

        action_mean, value = self.get_action_and_value(state)

        dist = self.get_action_dist(action_mean)

        action_logprobs = torch.sum(dist.log_prob(action), dim=-1, keepdim=True)
        dist_entropy = torch.sum(dist.entropy(), dim=-1, keepdim=True)

        return action_logprobs, value, dist_entropy


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from plotting.utils import (
        visualize_gcn_relationships_over_time,
        visualize_gcn_relationships_static,
        visualize_attention_weights,
        visualize_attention_weights_over_time,
        visualize_edge_importance_over_time,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    graph_type = "chain"

    model = ActorCritic(
        n_agents_train=8,
        n_agents_eval=8,
        d_state=24,
        d_action=2,
        device=device,
        graph_type=graph_type,
    ).to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)

    # Get graph list and create batched graph
    state_batch = torch.randn(1, 8, 18).to(device)  # 8 nodes, 18-d features
    graph_list = create_graph_batch(state_batch, model.graph_type, model.device)
    batched_graph = Batch.from_data_list(graph_list)

    # Generate a sequence of states over time (simulation or real data)
    state_sequence = []
    for t in range(20):  # 20 timesteps
        # Either use recorded states or create simulated ones
        state = torch.randn(1, 8, 18).to(device)  # 8 nodes, 18-d features
        state_sequence.append(state)

    # Create static visualization
    # fig = visualize_gcn_relationships_static(model, state_sequence, num_samples=5)
    # visualize_attention_weights(model, state_batch)
    # visualize_attention_weights_over_time(model, state_sequence)

    visualize_edge_importance_over_time(
        model, state_sequence, node_idx=1
    )  # Focus on node 1

    plt.show()