import json
# from Replay_Buffer import ReplayBuffer
from Network import DQNetwork
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import mixed_precision
import os


class Agent(object):
    def __init__(self,
                 update_freq: int,
                 input_dims: tuple,
                 layer1_nodes: int,
                 epsilon: float,
                 n_actions: int,
                 lr: float,
                 gamma: float,
                 eps_decay: float,
                 eps_final: float,
                 mem_size: int,
                 batch_size: int,
                 q_eval_filename: str,
                 q_target_filename: str,
                 ):

        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

        self.q_eval_filename = q_eval_filename
        self.q_target_filename = q_target_filename
        self.action_space = [i for i in range(n_actions)]
        self.input_dims = input_dims
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_final = eps_final
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.learn_step = 0
        self.target_update_freq = update_freq
        self.layer1_nodes = layer1_nodes

        self.mem_size = 2500
        self.mem_cntr = 0
        self.batch_size = batch_size

        self.state_memories = np.zeros((2, self.mem_size, *input_dims),  # 2 for cuurent and next state memory
                                       dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        self.Q_eval = DQNetwork(lr=self.lr,
                                n_actions=n_actions,
                                input_dims=self.input_dims,
                                fc1_dims=self.layer1_nodes)

        self.Q_target = DQNetwork(lr=self.lr,
                                  n_actions=n_actions,
                                  input_dims=self.input_dims,
                                  fc1_dims=self.layer1_nodes)

    def store_transition(self, state, action, reward, state_, done):
        # print('state',state.shape)
        # print('state',state_.shape)
        index = self.mem_cntr % self.mem_size
        self.state_memories[0][index] = state
        self.state_memories[1][index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    os.environ['TF_DATA_EXPERIMENTAL_SLACK'] = '0'

    def replace_target_network(self):
        # print(self.learn_step)
        if self.learn_step != 0 and (self.learn_step % self.target_update_freq) == 0:
            self.Q_target.set_weights(self.Q_eval.get_weights())

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
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
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_size, self.mem_cntr)
        batch = np.random.choice(max_mem, size=self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memories[0][batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.state_memories[1][batch]).to(self.Q_eval.device)
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

        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        self.epsilon = max((self.epsilon - self.eps_decay),
                           self.eps_final)  # we used linear decay because it takes more iterations to reach null value

    #
    def save_models(self, q_eval_path,q_target_path):
        try:
            T.save(self.Q_eval.state_dict(), q_eval_path)
            T.save(self.Q_target.state_dict(), q_target_path)
            print('Models saved successfully!')
        except:
            print('Error saving models ')

    def load_models(self, q_eval_path, q_target_path):
        try:
            self.Q_eval = T.load(q_eval_path, map_location=self.Q_eval.device)
            self.Q_eval = T.load(q_target_path, map_location=self.Q_target.device)
        except FileNotFoundError:
            print(f"Error: Could not find model files at {q_eval_path}")
        except RuntimeError as e:
            print(f"Error loading model: {e}")
