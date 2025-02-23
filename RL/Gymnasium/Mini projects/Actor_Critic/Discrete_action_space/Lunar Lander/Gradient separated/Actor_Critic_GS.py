import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import os, pickle


class Actor(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=4):
        super(Actor, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.action_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        x = F.relu(self.fc(state))
        action_probs = F.softmax(self.action_layer(x), dim=-1)
        return action_probs


class Critic(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.value_layer = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc(state))
        state_value = self.value_layer(x)
        return state_value