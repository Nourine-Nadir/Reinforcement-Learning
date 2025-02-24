import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical

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


class DiscreteActor(nn.Module):
    def __init__(self,
                 lr,
                 min_lr,
                 input_shape,
                 fc1_dims,
                 fc2_dims,
                 n_actions,
                 action_space=None):
        super(DiscreteActor, self).__init__()
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
        self.fc1 = nn.Linear(*self.input_shape, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions )
        #  Ensure weights initialization
        self.apply(weights_init_)

        #  Optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(0.99 ** epoch, (self.min_lr / self.lr))
        )

        #  Device selection
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)



    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float32, device=self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=-1)

        return x

    def sample(self, observation):
        #  Get mean and standard deviation
        probs = self(observation)
        action_distribution = Categorical(probs)
        action = action_distribution.sample()

        log_probs = action_distribution.log_prob(action)


        return action, log_probs, action_distribution

    def lr_decay(self):
        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]
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
        self.fc1 = nn.Linear(*self.input_shape, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions * 2)
        #  Ensure weights initialization
        self.apply(weights_init_)

        #  Optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(0.999 ** epoch, (self.min_lr / self.lr))
        )

        #  Device selection
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
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
            #  Change of variable theorem : log p(tanh(x)) = log p(x) - log(d/dx tanh(x))
            #  The derivative of tanh(x) is (1 - tanh²(x))
            #  Therefore, we subtract log(1 - tanh²(x)) from our original log probability
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


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
class GaussianPolicy(nn.Module):
    def __init__(self,
                 lr,
                 min_lr,
                 input_shape,
                 fc1_dims,
                 fc2_dims,
                 n_actions,
                 action_space=None):
        super(GaussianPolicy, self).__init__()
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
            #  Hidden Layers
        self.fc1 = nn.Linear(*self.input_shape, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
            #  Output Layers
        self.mean_linear = nn.Linear(self.fc2_dims, self.n_actions)
        self.log_std_linear = nn.Linear(self.fc2_dims, self.n_actions)

        #  Ensure weights initialization
        self.apply(weights_init_)

        #  Optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = T.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000, eta_min=self.min_lr)


        #  Device selection
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

        #  Get action scales
        self.action_scale, self.action_bias = action_scaling(self.action_space)
        self.action_scale = self.action_scale.to(self.device)
        self.action_bias = self.action_bias.to(self.device)

    def forward(self, obs):
        state = T.tensor(obs, dtype=T.float32, device=self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = T.clamp(log_std, min =LOG_SIG_MIN, max =LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):
        mu, sigma = self.forward(state)

        std = T.exp(sigma)
        dist = Normal(mu, std)
        probs = dist.rsample()

        y_t = T.tanh(probs)

        action = y_t * self.action_scale + self.action_bias
        log_probs = dist.log_prob(probs).sum(dim=-1).to(self.device) # Sum across action dimensions

        log_probs -= T.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON)

        log_probs = log_probs.sum()

        mean = mu * self.action_scale + self.action_bias # The mean of a Gaussian distribution doesn’t need rescaling.

        return action, log_probs, mu # Use mu directly, not tanh(mu)

    def lr_decay(self):
        self.scheduler.step()

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]


class QNetwork(nn.Module):
    def __init__(self,
                 lr,
                 min_lr,
                 input_shape,
                 n_actions,
                 hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(input_shape + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(input_shape + n_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

        #  Optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(0.999 ** epoch, (min_lr / lr))
        )
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        xu = T.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]

class ValueNetwork(nn.Module):
    def __init__(self,
                 lr,
                 min_lr,
                 input_shape,
                 fc1_dims,
                 fc2_dims
                 ):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(input_shape, fc1_dims)
        self.linear2 = nn.Linear(fc1_dims, fc2_dims)
        self.linear3 = nn.Linear(fc2_dims, 1)

        self.apply(weights_init_)

        #  Optimizer
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = T.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: max(0.99 ** epoch, (min_lr / lr))
        )
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        value = self.linear3(x)
        return value

    def get_lr(self):
        return self.scheduler.get_last_lr()[0]

    def lr_decay(self):
        self.scheduler.step()


