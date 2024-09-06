import numpy as np
import matplotlib.pyplot as plt
from Custom_Env import CustomFrozenLake
from DQN_agent import Agent
import pickle,json,os
from tqdm import tqdm
import gymnasium as gym

with open('params.json', 'r') as f:
    params = json.load(f)['parameters']
class Experiment():
    def __init__(self,group=1):
        self.env = CustomFrozenLake(map_name="8x8", is_slippery=False, render_mode='rgb_array')

        self.learning_rate = params['learning_rate']
        self.n_episodes = params['n_episodes']
        self.initial_epsilon = params['initial_eps']
        self.epsilon_decay = params['eps_decay']
        self.final_epsilon = params['final_eps']
        self.agent_path = params['agent_path']
        self.group = group
        self.agent = Agent(gamma=.99,
                      initial_eps=1.0,
                      eps_decay=1e-5,
                      final_eps=0.05,
                      batch_size=128,
                      n_actions=4,
                      input_dims=[1],
                      lr=0.0001,
                      max_mem_size=10_000)
#%%
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env, deque_size=self.n_episodes)
        self.env = gym.wrappers.TimeLimit(self.env, max_episode_steps=60)


    def Run(self):
        self.rewards = 0
        if self.group == 3 or self.group == 2 :
            self.env.unwrapped.set_goalReward(0)
            self.env.unwrapped.set_stepPenalty(-1)
            self.env.unwrapped.set_stuckPenalty(-1)

        for episode in tqdm(range(self.n_episodes)):
            score = 0
            done = False
            obs, info = self.env.reset()

            while not done:
                action = self.agent.choose_action(obs)
                obs_, reward, terminated, truncated, info = self.env.step(action)
                score += reward
                self.agent.store_transition(obs, action, reward, obs_, done)
                self.agent.learn()
                if self.group ==3:
                    if episode ==400 :
                        self.env.unwrapped.set_goalReward(100)
                        self.env.unwrapped.set_stepPenalty(-1)
                        self.env.unwrapped.set_stuckPenalty(-1)
                obs = obs_
                done = terminated or truncated



        self.agent.save_agent(self.agent_path+'DQN_group '+str(self.group))


    def Plot_save(self,rolling_length=50):

        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
        axs[0].set_title("Episode rewards")
        # compute and assign a rolling average of the data to provide a smoother graph
        reward_moving_average = (
            np.convolve(
                np.array(self.env.get_wrapper_attr('return_queue')).flatten(), np.ones(rolling_length), mode="valid"
            )
            / rolling_length
        )
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[1].set_title("Episode lengths")
        length_moving_average = (
            np.convolve(
                np.array(self.env.get_wrapper_attr('length_queue')).flatten(), np.ones(rolling_length), mode="same"
            )
            / rolling_length
        )

        plt.tight_layout()

        if not os.path.exists('figs/DQN'):
            os.makedirs('figs/DQN')

        try:
            plt.savefig('figs/DQN/group nb ' + str(self.group) + '.png')
            print('File saved correctly')
        except:
            print('Error saving figure')