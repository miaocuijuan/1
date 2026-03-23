"""Network modules"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = MLP(state_dim, hidden_dim, action_dim)

    def forward(self, state):
        return torch.softmax(self.net(state), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = MLP(state_dim + action_dim, hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
