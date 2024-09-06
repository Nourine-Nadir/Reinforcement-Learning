import numpy as np
import matplotlib.pyplot as plt
from Custom_Env import CustomFrozenLake
from Q_value_Agent import Agent
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
        self.agent = Agent(
            learning_rate = self.learning_rate,
            initial_epsilon = self.initial_epsilon,
            epsilon_decay = self.epsilon_decay,
            final_epsilon = self.final_epsilon,
            env = self.env
        )
#%%
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env, deque_size=self.n_episodes)
        self.env = gym.wrappers.TimeLimit(self.env, max_episode_steps=60)


    def Run(self):
        self.rewards = 0
        if self.group == 3 or self.group == 2 :
            self.env.unwrapped.set_goalReward(0)
            self.env.unwrapped.set_stepPenalty(0)
            self.env.unwrapped.set_stuckPenalty(-1)

        for episode in tqdm(range(self.n_episodes)):

            obs, info = self.env.reset()
            done = False

            # play one episode
            while not done:
                action = self.agent.choose_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.rewards += reward
                # update the agent
                self.agent.update_q_values(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                if self.group ==3:
                    if episode ==800 :
                        self.env.unwrapped.set_goalReward(100)
                        self.env.unwrapped.set_stepPenalty(-1)
                        self.env.unwrapped.set_stuckPenalty(-1)
                done = terminated or truncated
                obs = next_obs

            self.agent.decay_epsilon()

        self.agent.save_agent(self.agent_path+'Q_Table group '+str(self.group))


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
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        axs[2].set_title("Training Error")
        training_error_moving_average = (
            np.convolve(np.array(self.agent.training_error), np.ones(rolling_length), mode="same")
            / rolling_length
        )
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
        plt.tight_layout()

        if not os.path.exists('figs/Q-Table'):
            os.makedirs('figs/Q-Table')
        try:
            plt.savefig('figs/Q-Table/group nb '+str(self.group)+ '.png')
            print('File saved correctly')
        except:
            print('Error saving figure')