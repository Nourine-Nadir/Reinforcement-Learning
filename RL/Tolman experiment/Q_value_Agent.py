from collections import defaultdict
import numpy as np
import pickle,os

class PicklableDefaultDict(defaultdict):
    def __init__(self, default_factory=None):
        super().__init__(default_factory)

    def __reduce__(self):
        return type(self), (self.default_factory,), None, None, iter(self.items())

class Agent():
    def __init__(self,
                 learning_rate: float,
                 initial_epsilon: float,
                 epsilon_decay: float,
                 final_epsilon: float,
                 env,
                 discount_factor: float = 0.95,
                 ):

        self.env = env
        self.q_values = PicklableDefaultDict(self.default_q_value)

        self.lr = learning_rate
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.discount_factor = discount_factor

        self.training_error = []

    def default_q_value(self):
        return np.zeros(self.env.action_space.n)

    def choose_action(self, obs: tuple[int, int, bool]) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update_q_values(self,
                        obs: tuple[int, int, bool],
                        action: int,
                        reward: float,
                        terminated: bool,
                        next_obs: tuple[int, int, bool]):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        temporal_diffrence = (reward + (self.discount_factor * future_q_value)) - self.q_values[obs][action]

        self.q_values[obs][action] = (
                self.q_values[obs][action] + self.lr * temporal_diffrence
        )
        self.training_error.append(temporal_diffrence)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def save_agent(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(path, 'wb') as f:
            pickle.dump(self, f)
    def load_agent(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
