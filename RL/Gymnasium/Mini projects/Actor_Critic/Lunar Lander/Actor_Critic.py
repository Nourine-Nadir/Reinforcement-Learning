import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.affine = nn.Linear(8, 128)

        self.action_layer = nn.Linear(128, 4)
        self.value_layer = nn.Linear(128, 1)


        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.action_distributions = []
        self.alpha = torch.tensor(0.99)

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.affine(state))

        state_value = self.value_layer(state)

        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        self.logprobs.append(action_distribution.log_prob(action))
        self.action_distributions.append(action_distribution)
        self.state_values.append(state_value)

        return action.item()

    def calculateLoss(self, gamma=0.99):

        # calculating discounted rewards:
        rewards = []
        dis_reward = 0
        for reward in self.rewards[::-1]:
            dis_reward = reward + gamma * dis_reward
            rewards.insert(0, dis_reward)

        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())

        loss = 0
        for logprob, value, reward, action_distribution in zip(self.logprobs, self.state_values, rewards, self.action_distributions):
            advantage = reward - value.item()
            action_loss = -logprob * advantage
            value_loss = F.smooth_l1_loss(value, reward)
            entropy = torch.tensor(action_distribution.entropy())
            loss += (action_loss + value_loss - self.alpha * entropy)

        self.alpha =  torch.tensor(max((self.alpha.item() - 1e-3), -1))
        return loss

    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
