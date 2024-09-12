import json
from Replay_Buffer import ReplayBuffer
from Network import DQNetwork
from keras.optimizers import Adam
from keras.models import load_model
import tensorflow as tf
import random
import numpy as np

with open('params.json', 'r') as f:
    params = json.load(f)["parameters"]

num_games, gamma, initial_eps, eps_decay, \
    final_eps, batch_size, n_actions, input_dims, \
    lr, max_memory_size, model_path, Q_eval_path, Q_target_path, agent_path, \
    layer1_dims, layer2_dims, layer3_dims, update_freq \
    = \
    (params[key] for key in
     list(params.keys())
     )
input_dims = (input_dims['nb_images'], input_dims['height'], input_dims['width'])
class Agent(object):
    def __init__(self,
                 update_freq=update_freq,
                 input_dims=input_dims,
                 epsilon=initial_eps,
                 n_actions=n_actions,
                 lr=lr,
                 gamma=gamma,
                 eps_decay: float = eps_decay,
                 eps_final: float = final_eps,
                 mem_size=max_memory_size,
                 batch_size=batch_size,
                 q_eval_filename=Q_eval_path,
                 q_target_filename=Q_eval_path,
                 ):
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
        self.target_update_freq = 100

        self.memory = ReplayBuffer(max_size=self.mem_size,
                                   input_shape=self.input_dims)
        with tf.device('/GPU:0'):

            self.Q_eval = DQNetwork(lr=self.lr,
                                    n_actions=n_actions,
                                    input_dims=self.input_dims,
                                    fc1_dims=layer1_dims)
            self.Q_eval.compile(optimizer=Adam(learning_rate=lr), loss='mse')

            self.Q_target = DQNetwork(lr=self.lr,
                                      n_actions=n_actions,
                                      input_dims=self.input_dims,
                                      fc1_dims=layer1_dims)
            self.Q_target.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    def replace_target_network(self):
        if self.target_update_freq != 0 and self.learn_step % self.target_update_freq == 0:
            self.Q_target.set_weights(self.Q_eval.get_weights())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def choose_action(self, obs):
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            print([obs])
            state = np.array([obs], copy=False, dtype=np.float32)
            action = np.argmax(self.Q_eval.predict(state,verbose=0))

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, next_state, done = \
                self.memory.sample_buffer(self.batch_size)

            self.replace_target_network()
            with tf.device('/GPU:0'):

                q_eval = self.Q_eval.predict(state,verbose=0)
                q_next = self.Q_target.predict(next_state,verbose=0)

            q_next[done] = 0.0

            indices = np.arange(self.batch_size)
            q_target = q_eval.copy()

            q_target[indices, action] = reward + \
                                        self.gamma * np.max(q_next, axis=1)
            with tf.device('/GPU:0'):

                self.Q_eval.train_on_batch(state,
                                           q_target)  # This will apply a forward pass with state as input and compute mse loss with q_target to backprop

            self.epsilon = max(self.epsilon - self.eps_decay, final_eps)
            self.learn_step += 1

    def save_models(self):
        try:
            self.Q_eval.save(self.q_eval_filename)
            self.Q_target.save(self.q_target_filename)
            print("Model saved successfully !")
        except:
            print("Model could not be saved")

    def load_models(self):
        try:
            self.Q_eval = load_model(self.q_eval_filename)
            self.Q_target = load_model(self.q_target_filename)
            print("Model loaded successfully !")
        except:
            print("Model could not be loaded ")
