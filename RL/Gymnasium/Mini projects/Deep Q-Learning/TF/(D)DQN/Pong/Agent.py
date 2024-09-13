import json
from Replay_Buffer import ReplayBuffer
from Network import DQNetwork
from keras.optimizers import Adam
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras import mixed_precision
import os

class Agent(object):
    def __init__(self,
                 update_freq:int,
                 input_dims:tuple,
                 layer1_nodes:int,
                 epsilon:float,
                 n_actions:int,
                 lr:float,
                 gamma:float,
                 eps_decay: float ,
                 eps_final: float,
                 mem_size:int,
                 batch_size:int,
                 q_eval_filename:str,
                 q_target_filename:str,
                 ):
        physical_devices = tf.config.list_physical_devices('GPU')
        os.environ['TF_DATA_EXPERIMENTAL_SLACK'] = '0'

        if physical_devices:
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except RuntimeError as e:
                print(e)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)



        self.optimizer = Adam(learning_rate=lr)
        self.loss_fn = keras.losses.MeanSquaredError()

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        print('tf version : ', tf.__version__)
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

        self.memory = ReplayBuffer(max_size=self.mem_size,
                                   input_shape=self.input_dims)

        with tf.device('/GPU:0'):

            self.Q_eval = DQNetwork(lr=self.lr,
                                    n_actions=n_actions,
                                    input_dims=self.input_dims,
                                    fc1_dims=self.layer1_nodes)
            self.Q_eval.compile(optimizer=Adam(learning_rate=lr), loss='mse')

            self.Q_target = DQNetwork(lr=self.lr,
                                      n_actions=n_actions,
                                      input_dims=self.input_dims,
                                      fc1_dims=self.layer1_nodes)
            self.Q_target.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    os.environ['TF_DATA_EXPERIMENTAL_SLACK'] = '0'

    def replace_target_network(self):
        if self.learn_step != 0 and (self.learn_step % self.target_update_freq) == 0:
            self.Q_target.set_weights(self.Q_eval.get_weights())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def choose_action(self, obs):
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            state = np.array([obs], copy=False, dtype=np.float32)
            action = np.argmax(self.Q_eval.predict(state,verbose=0))

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, next_state, done = \
                self.memory.sample_buffer(self.batch_size)
            self.replace_target_network()

            with tf.device('/GPU:0'):
                q_eval = self.Q_eval.predict(state, verbose=0)
                q_next = self.Q_target.predict(next_state, verbose=0)

            q_next[done] = 0.0

            indices = np.arange(self.batch_size)
            q_target = q_eval.copy()

            q_target[indices, action] = reward + \
                                        self.gamma * np.max(q_next, axis=1)

            # -------------------------------------------
            loss_fn = keras.losses.MeanSquaredError()
            optimizer = keras.optimizers.Adam()

            with tf.device('/GPU:0'):
                with tf.GradientTape() as tape:
                    logits = self.Q_eval(state, training=True)  # forward pass
                    train_loss_value = loss_fn(q_target, logits)  # compute loss

                grads = tape.gradient(train_loss_value, self.Q_eval.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.Q_eval.trainable_weights))


    def update_epsilon(self):
        #Moved these changement here because it can't run on the graph mode inside a %tf.function
        self.epsilon = max(self.epsilon - self.eps_decay, self.eps_final)
        self.learn_step += 1

    def save_models(self):
        try:
            dummy_input = tf.zeros((1, *self.input_dims))
            _ = self.Q_eval(dummy_input)
            _ = self.Q_target(dummy_input)


            tf.saved_model.save(self.Q_eval, self.q_eval_filename)
            tf.saved_model.save(self.Q_target, self.q_target_filename)
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
