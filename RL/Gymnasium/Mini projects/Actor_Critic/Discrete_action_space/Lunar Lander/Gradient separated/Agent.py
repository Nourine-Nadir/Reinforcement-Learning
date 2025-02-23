import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import os, pickle
from Actor_Critic_GS import *
class Agent:
    def __init__(self,
                 input_dim=8,
                 hidden_dim=128,
                 output_dim=4,
                 gamma=0.99,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 eps_clip=0.2,
                 update_freq=100):

        self.gamma = gamma
        self.eps_clip = eps_clip  # Clipping for PPO (optional)
        self.update_freq = update_freq

        # Separate actor and critic
        self.actor = Actor(input_dim, hidden_dim, output_dim)
        self.critic = Critic(input_dim, hidden_dim)

        # Separate optimizers for GSAC
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Memory for training
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.states = []

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        action_probs = self.actor(state)
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()

        # Store log prob and state value
        self.logprobs.append(action_distribution.log_prob(action))
        self.state_values.append(self.critic(state))  # No detach for proper backprop
        self.states.append(state)

        return action.item()

    def calculate_loss(self):
        rewards = []
        discounted_reward = 0
        for reward in reversed(self.rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / rewards.std() +1e-8

        actor_loss = 0
        critic_loss = 0
        for logprob, value, reward in zip(self.logprobs, self.state_values, rewards):
            advantage = reward - value.item()

            # Actor loss (policy gradient)
            actor_loss += -logprob * advantage.detach()

            # Critic loss (value function update)
            critic_loss += F.smooth_l1_loss(value, torch.tensor([reward]))

        return actor_loss, critic_loss

    def learn(self):
        actor_loss, critic_loss = self.calculate_loss()

        # Update actor separately
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic separately
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.clear_memory()

    def clear_memory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
        del self.states[:]

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict()}, path)

    def load_model(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
        else:
            print(f"Error: Model file not found at {path}")