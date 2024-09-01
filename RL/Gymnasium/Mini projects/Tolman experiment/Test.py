from Q_value_Agent import Agent
from Custom_Env import CustomFrozenLake
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you prefer Qt

import pickle,json
import gymnasium as gym

import numpy as np



with open('params.json', 'r') as f:
    params = json.load(f)['parameters']


agent_path = params['agent_path']+ ' group'+ str(1)
with open(agent_path, 'rb') as f:
    agent = pickle.load(f)

env = CustomFrozenLake(map_name="8x8", is_slippery=False, render_mode='rgb_array')
obs, info = env.reset()

plt.ion()
fig, ax = plt.subplots(figsize=(8 ,8))
action_text = ax.text(510, 20, '', color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.8))
img = ax.imshow(env.render())
actions = ['Move Up' ,'Move Right' ,'Move Down' ,'Move Left']
rewards = 0
num_epochs= 100
for step in range(num_epochs):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.choose_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        rewards += reward

        print(f'step {step}:  obs = {next_obs} , reward = {reward}')
        frame = env.render()
        img.set_data(frame)
        action_text.set_text(f'Step: {actions[action] }')

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(.05)
        done = terminated or truncated
        obs = next_obs

plt.ioff()  # Turn off interactive mode
# plt.show()  # Keep the window open after the animation finishes
plt.close()
env.close()