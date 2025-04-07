import os
import numpy as np
import torch as T
import torch.nn.functional as F
from tqdm import tqdm
from replay_memory import ReplayBuffer
from models import *
from utils import soft_update, hard_update


class Agent():
    def __init__(self,
                 input_shape,
                 fc1_dims: int,
                 fc2_dims: int,
                 action_space,
                 epochs: int,
                 lr: float = 1e-3,
                 min_lr: float = 1e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 alpha_clip: float = 0.2,
                 batch_size: int = 256,
                 entropy_factor : float = 0.995
                 ):
        # Initialize attributes
        #  Model dims
        self.action_space = action_space
        self.n_actions = action_space.shape[0]
        self.input_shape = input_shape
        self.epochs = epochs

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims

        #  Model hyperparameters
        self.lr = lr
        self.min_lr = min_lr

        #  PPO hyperparameters
        self.alpha = alpha_clip  # Policy clip
        self.gamma = gamma  # Reward decay
        self.gae_lambda = gae_lambda  # the factor of gamma decay
        self.entropy_factor = entropy_factor
        # Replay Buffer
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(self.batch_size)

        # --------- PRINTING --------
        print(f'input shape : {self.input_shape[0]}'
              f' \naction_space : {self.action_space}'
              f' \n lr : {self.lr}'
              f' \nfc1_dims : {self.fc1_dims}'
              f' \nfc2_dims : {self.fc2_dims}'
              f' \ngamma : {self.gamma}'
              f' \nGAE lambda: {self.gae_lambda}'
              f' \nepochs: {self.epochs}'
              f' \nbatch_size : {self.batch_size} ')

        # Actor & Critic models
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.policy = GaussianPolicy(lr=self.lr,
                                    min_lr=self.min_lr,
                                    input_shape=self.input_shape,
                                    fc1_dims=self.fc1_dims,
                                    fc2_dims=self.fc2_dims,
                                    n_actions=self.n_actions ,
                                    action_space=self.action_space)
        self.Critic = GenericNetwork(lr=self.lr,
                                     min_lr=self.min_lr,
                                     input_shape=self.input_shape,
                                     fc1_dims=self.fc1_dims,
                                     fc2_dims=self.fc2_dims,
                                     n_actions=1,
                                     action_space=self.action_space)

    # Store transition
    def store_transition(self, state, action, log_prob, value, reward, done):
        # Convert to CPU tensors before storing
        if isinstance(log_prob, T.Tensor):
            log_prob = log_prob.detach().cpu()
        if isinstance(value, T.Tensor):
            value = value.detach().cpu()

        self.buffer.push(state, action, log_prob, value, reward, done)

    def clearMemory(self):
        self.buffer.clear()
        self.log_probs_ = []

    # Sample action from Actor network and State value to return it and store it in the buffer after
    def choose_action(self, obs):
        #
        action, log_probs, _ = self.policy.sample(obs)

        obs = T.from_numpy(obs).float().to(self.device)
        value = self.Critic(obs).squeeze(0)

        return [action.item()], log_probs, value

    def normalize_rewards(self, rewards):
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return rewards

    # Run gradient on both Actor and Critic networks
    def update_parameters(self):
        for _ in range(self.epochs):

            # Sample shuffled mini-batches
            state_arr, action_arr, old_prob_arr, vals_arr, \
                reward_arr, dones_arr, batches = \
                self.buffer.sample()

            # Transform np.arrays to tensors
            values = T.tensor(vals_arr, device=self.device).detach()
            action_arr = T.tensor(action_arr, device=self.device).detach()
            old_prob_arr = T.tensor(old_prob_arr, device=self.device).detach()
            reward_arr = T.tensor(reward_arr, device=self.device).detach()
            dones_arr = T.tensor(dones_arr, device=self.device).detach()

            # Compute advantages using GAE
            advantage = self.compute_advantages(reward_arr, dones_arr, values)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)  # Normalize

            # Compute returns (TD target)
            returns = advantage + values  # No detach() here, since it's part of the loss

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device)
                old_probs = old_prob_arr[batch].to(self.device)
                actions = action_arr[batch].to(self.device)

                _, _, dist = self.policy.sample(states)
                critic_value = self.Critic(states).squeeze()

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.alpha,
                                                 1 + self.alpha) * advantage[batch]
                entropy = dist.entropy().mean()
                policy_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                # Clipped value loss
                value_pred_clipped = values[batch] + (critic_value - values[batch]).clamp(-self.alpha, self.alpha)
                vf_loss1 = F.mse_loss(critic_value.squeeze(), returns[batch].squeeze())
                vf_loss2 = F.mse_loss(value_pred_clipped.squeeze(), returns[batch].squeeze())
                vf_loss = T.max(vf_loss1, vf_loss2).mean()

                total_loss = policy_loss + 1 * vf_loss - self.entropy_factor * entropy


                self.policy.optimizer.zero_grad()
                self.Critic.optimizer.zero_grad()
                total_loss.backward()
                self.policy.optimizer.step()
                self.Critic.optimizer.step()

        self.buffer.clear()
        self.policy.lr_decay()
        self.Critic.lr_decay()
    # Compute advantages
    def compute_advantages(self, rewards, mask, values):
        # Make sure the tensor types are consistent
        rewards = rewards.float()  # Ensure float32 not float64
        mask = mask.float()  # Ensure float32
        values = values.float()  # Ensure float32

        advantages = T.zeros_like(rewards, device=self.device)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # Terminal state has no future value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - mask[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - mask[t]) * last_gae
            advantages[t] = last_gae

        return advantages

    def get_lr(self):
        return self.policy.get_lr(), self.Critic.get_lr()

    def save_model(self, PATH):
        # Verify if the directory exists
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        # Save the model
        T.save(self.policy.state_dict(), PATH)

    def load_model(self, PATH):
        try:
            print('imported model')
            self.policy.load_state_dict(T.load(PATH, weights_only=True))
        except FileNotFoundError:
            print(f"Error: Could not find model files at {PATH}")
        except RuntimeError as e:
            print(f"Error loading model: {e}")
