import random
import numpy as np
import torch as T


class ReplayBuffer():
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def sample(self):
        # For PPO, we don't want to shuffle - we need to maintain trajectory order
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        # DO NOT SHUFFLE for on-policy algorithms
        # np.random.shuffle(indices)  <- Remove this line
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        # Convert lists to numpy arrays with appropriate data types
        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions, dtype=np.float32)
        probs = np.array(self.probs, dtype=np.float32)
        vals = np.array(self.vals, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        return states, actions, probs, vals, rewards, dones, batches

    def push(self, state, action, probs, vals, reward, done):
        # Store each element separately for better control
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def __len__(self):
        return len(self.states)
