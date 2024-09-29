import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyGradient(nn.Module):
    def __init__(self, alpha, final_alpha):
        super(PolicyGradient, self).__init__()
        self.affine = nn.Linear(8, 128)

        self.action_layer = nn.Linear(128, 4)
        self.value_layer = nn.Linear(128, 1)


        self.logprobs = []
        self.rewards = []
        self.action_distributions = []
        self.alpha = torch.tensor(alpha)
        self.final_alpha = final_alpha

    def forward(self, state):
        state = torch.from_numpy(state).float()
        state = F.relu(self.affine(state))


        action_probs = F.softmax(self.action_layer(state))
        action_distribution = Categorical(action_probs)
        action = action_distribution.sample()
        self.logprobs.append(action_distribution.log_prob(action))
        self.action_distributions.append(action_distribution)

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
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        policy_loss = []
        for logprob, reward, action_distribution in zip(self.logprobs, rewards, self.action_distributions):

            entropy = torch.tensor(action_distribution.entropy())
            policy_loss.append((-logprob * reward) - (self.alpha * entropy))
        policy_loss = torch.stack(policy_loss).sum()
        self.alpha =  torch.tensor(max((self.alpha.item() * .995), self.final_alpha))
        return policy_loss

    def clearMemory(self):
        del self.logprobs[:]
        del self.rewards[:]
