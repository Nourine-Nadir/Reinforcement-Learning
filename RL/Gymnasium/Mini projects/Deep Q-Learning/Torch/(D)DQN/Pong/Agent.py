import json
# from Replay_Buffer import ReplayBuffer
from Network import DQNetwork
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

import os, pickle


class Agent(object):
    def __init__(self,
                 update_freq: int,
                 input_shape: tuple,
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


        self.update_freq = update_freq
        self.input_shape = input_shape
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.eps_final = eps_final
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.q_eval_filename = q_eval_filename
        self.q_target_filename = q_target_filename

        self.action_space = [i for i in range(n_actions)]
        self.learn_step = 0
        self.mem_cntr = 0

        self.state_memories = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memories = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.Q_eval = DQNetwork(lr=self.lr, n_actions=n_actions, input_shape=input_shape)
        self.Q_target = DQNetwork(lr=self.lr, n_actions=n_actions, input_shape=input_shape)

        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_target.eval()

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memories[index] = state
        self.new_state_memories[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = T.tensor(np.array([observation]), dtype=T.float32).to(self.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_size, self.mem_cntr)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = T.tensor(self.state_memories[batch]).to(self.device)
        new_state_batch = T.tensor(self.new_state_memories[batch]).to(self.device)
        action_batch = T.tensor(self.action_memory[batch]).to(self.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.device)

        q_eval = self.Q_eval.forward(state_batch)[T.arange(self.batch_size), action_batch]
        q_next = self.Q_target.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.update_freq == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())

        self.epsilon = max(self.epsilon - self.eps_decay, self.eps_final)

    def save_models(self):
        T.save(self.Q_eval.state_dict(), self.q_eval_filename)
        T.save(self.Q_target.state_dict(), self.q_target_filename)
        print("Models saved successfully!")

    def load_models(self):
        try:
            self.Q_eval.load_state_dict(T.load(self.q_eval_filename, map_location=self.device))
            self.Q_target.load_state_dict(T.load(self.q_target_filename, map_location=self.device))
            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading Pong models: {e}")
            print("Initializing new Pong models.")
            self.Q_eval = DQNetwork(lr=self.lr, n_actions=self.n_actions, input_shape=self.input_shape)
            self.Q_target = DQNetwork(lr=self.lr, n_actions=self.n_actions, input_shape=self.input_shape)
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
            self.Q_target.eval()
