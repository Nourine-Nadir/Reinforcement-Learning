import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

EPSILON = 1e-6


def weights_init_(m):
    if isinstance(m, nn.Linear):
        T.nn.init.xavier_uniform_(m.weight, gain=1)
        T.nn.init.constant_(m.bias, 0)


def action_scaling(action_space):
    if action_space == None:
        action_scale = T.tensor(1.)
        action_bias = T.tensor(0.)
    else:
        action_scale = T.tensor(
            (action_space.high - action_space.low) / 2,
            dtype=T.float32)
        action_bias = T.tensor(
            (action_space.high + action_space.low) / 2,
            dtype=T.float32)

        return action_scale, action_bias


class GenericNetwork(nn.Module):
    def __init__(self,
                 lr,
                 min_lr,
                 input_shape,
                 fc1_dims,
                 fc2_dims,
                 n_actions,
                 action_space=None):
        super(GenericNetwork, self).__init__()
        #  Layers dimension
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        #  Learning rates
        self.lr = lr
        self.min_lr = min_lr

        #  Inputs & outputs dims
        self.action_space = action_space
        self.input_shape = input_shape
        self.n_actions = n_actions

        #  Layers
        self.fc1 = nn.Linear(*self.input_shape[0], self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        #  Ensure weights initialization
        self.apply(weights_init_)

        #  Optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(0.9999 ** epoch, (self.min_lr / self.lr))
        )

        #  Device selection
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

        #  Get action scales
        self.action_scale, self.action_bias = action_scaling(self.action_space)
        self.action_scale = self.action_scale.to(self.device)
        self.action_bias = self.action_bias.to(self.device)

    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float32, device=self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def sample(self, observation, n_outputs):
        #  Get mean and standard deviation
        mu, sigma = self.forward(observation)

        std = T.exp(sigma)
        dist = Normal(mu, std)
        probs = dist.sample(sample_shape=T.Size([n_outputs]))
        log_probs = dist.log_prob(probs).to(self.device)

        y_t = T.tanh(probs)

        #  Apply the change of variables formula
        log_probs -= T.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)
        #  Sum across action dimensions
        log_probs = log_probs.sum()

        #  Scale actions
        action_scaled = y_t * self.action_scale + self.action_bias

        return action_scaled, log_probs

    def lr_decay(self):
        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]


