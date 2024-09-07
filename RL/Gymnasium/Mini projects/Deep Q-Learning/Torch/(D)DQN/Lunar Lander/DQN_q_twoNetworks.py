import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle,json,os

with open('params.json', 'r') as f:
    params = json.load(f)["parameters"]
fc1_dim = params['layer1_dims']
fc2_dim = params['layer2_dims']
fc3_dim = params['layer3_dims']
class DQNetwork(nn.Module):
    def __init__(self,
                 lr,
                 input_dims,
                 fc1_dims,
                 fc2_dims,
                 fc3_dims,
                 nb_actions):
        super(DQNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc2_dims
        self.nb_actions = nb_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims, self.nb_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.SmoothL1Loss()  # Use Huber loss
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        l1 = F.relu(self.fc1(state))
        l2 = F.relu(self.fc2(l1))
        # l3 = F.relu(self.fc3(l2))
        actions = self.fc4(l2)  # don't activate the last layer

        return actions


class Agent():
    def __init__(self,
                 batch_size: float,
                 input_dims,
                 n_actions: int,
                 max_mem_size: int = 10_000,
                 gamma: float = 0.99,
                 initial_eps: float = 1.0,
                 final_eps: float = 0.01,
                 eps_decay: float = 1e-4,
                 lr: float = 1e-4):
        self.gamma = gamma
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.action_space = [i for i in range(n_actions)]
        self.eps_decay = eps_decay
        self.lr = lr
        self.eps = initial_eps
        self.eps_decay = eps_decay
        self.final_eps = final_eps
        self.mem_size = max_mem_size
        self.mem_counter = 0
        self.sequence_length = 32  # You can adjust this value

        self.target_update_frequency = 100
        self.learn_step_counter = 0
        self.Q_eval = DQNetwork(lr=self.lr,
                                input_dims=self.input_dims,
                                nb_actions=n_actions,
                                fc1_dims=fc1_dim, fc2_dims=fc2_dim, fc3_dims=fc3_dim)
        self.Q_target = DQNetwork(lr=self.lr,
                                  input_dims=self.input_dims,
                                  nb_actions=n_actions,
                                  fc1_dims=fc1_dim, fc2_dims=fc2_dim, fc3_dims=fc3_dim)

        self.state_memory = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *self.input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size,
                                      dtype=np.int32)  # set of integers because our actions belong to a discrete space
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):  # sate_ : new state
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def choose_action(self, obs):
        if np.random.random() > self.eps:
            self.Q_eval.eval()  # Set to evaluation mode
            with T.no_grad():
                state = T.tensor(np.array([obs])).to(self.Q_eval.device)
                actions = self.Q_eval.forward(state)
            self.Q_eval.train()  # Set back to training mode
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_counter < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_size, self.mem_counter)
        batch = np.random.choice(max_mem, size=self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        self.Q_target.eval()  # Set to evaluation mode
        with T.no_grad():
            q_next = self.Q_target.forward(new_state_batch)
        self.Q_target.train()  # Set back to training mode

        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        self.eps = max((self.eps - self.eps_decay), self.final_eps) # we used linear decay because it takes more iterations to reach null value

    def save_model(self, PATH):
        T.save(self.Q_eval.state_dict(), PATH)

    def load_model(self, q_eval_path ):
        try:
            self.Q_eval = T.load(q_eval_path, map_location=self.Q_eval.device)
        except FileNotFoundError:
            print(f"Error: Could not find model files at {q_eval_path }")
        except RuntimeError as e:
            print(f"Error loading model: {e}")


    def save_agent(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f)