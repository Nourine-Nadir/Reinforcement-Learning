import os
import numpy as np
import torch as T
import torch.nn.functional as F
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
                 batch_size: int = 256
                 ):
        # Initialize attributes
        #  Model dims
        self.action_space = action_space
        self.n_actions = action_space.n
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
        self.policy = DiscreteActor(lr=self.lr,
                                    min_lr=self.min_lr,
                                    input_shape=self.input_shape,
                                    fc1_dims=self.fc1_dims,
                                    fc2_dims=self.fc2_dims,
                                    n_actions=(self.n_actions),
                                    action_space=self.action_space).to(self.device)
        self.Critic = ValueNetwork(lr=self.lr,
                                   min_lr=self.min_lr,
                                   input_shape=self.input_shape[0],
                                   fc1_dims=256,
                                   fc2_dims=128
                                   ).to(self.device)

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

        return action.item(), log_probs, value

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

            advantage = self.compute_advantages(reward_arr, dones_arr, values)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device).detach()
                old_probs = T.tensor(old_prob_arr[batch]).to(self.device).detach()
                actions = T.tensor(action_arr[batch]).to(self.device).detach()

                _, _, dist = self.policy.sample(states)
                critic_value = self.Critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.alpha,
                                                 1 + self.alpha) * advantage[batch]
                policy_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = T.square(returns - critic_value).mean()

                total_loss = policy_loss + 0.5 * critic_loss

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
        advantages = np.zeros(rewards.shape, dtype=np.float32)
        for t in range(len(rewards) - 1):
            discount = 1
            a_t = 0
            for k in range(t, len(rewards) - 1):
                a_t += discount * (rewards[k] + self.gamma * values[k + 1] * \
                                   (1 - int(mask[k])) - values[k])
                discount *= self.gamma * self.gae_lambda
            advantages[t] = a_t
        advantages = T.tensor(advantages, dtype=T.float32, device=self.device).unsqueeze(-1)
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
