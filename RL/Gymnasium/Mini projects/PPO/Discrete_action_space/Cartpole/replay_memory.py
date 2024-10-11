import random
import numpy as np
import torch as T


class ReplayBuffer():
    def __init__(self, batch_size):
        self.buffer = []
        self.position = 0

        self.batch_size = batch_size

    def sample(self):
        n_states = self.__len__()
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        # print(f'n_state {n_states}')
        # print(f'indices {indices}')
        np.random.shuffle(indices)
        # print(f'shuffled indices {indices}')
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        # print(f'batches {batches}')

        states, actions, probs, vals, rewards, dones = map(np.stack, zip(*self.buffer))

        return states, actions, \
            probs, vals, \
            rewards, dones, \
            batches

    def push(self, state, action, probs, vals, reward, done):
        self.buffer.append(None)
        self.buffer[self.position] = state, action, probs, vals, reward, done
        self.position += 1

    def clear(self):
        self.buffer = []
        self.position = 0

    def __len__(self):
        return len(self.buffer)
