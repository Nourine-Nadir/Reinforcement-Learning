import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you prefer Qt
import matplotlib.pyplot as plt
import gymnasium as gym
from DQN_q_eval import Agent
import json

with open('params.json', 'r') as f:
    params = json.load(f)["parameters"]
gamma = params["gamma"]
initial_epsilon = params["initial_eps"]
final_epsilon = params["final_eps"]
epsilon_decay = params["eps_decay"]
batch_size = params["batch_size"]
n_actions = params["n_actions"]
input_dims = params["input_dims"]
lr = params["lr"]
max_memory = params["max_memory_size"]
model_path = params["model_path"]


env = gym.make('LunarLander-v2', render_mode='rgb_array')
obs, info = env.reset()
plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
img = ax.imshow(env.render())

action_text = ax.text(510, 20, '', color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.8))
actions = ['nothing', 'Left', 'main', 'Right']
rewards = 0
num_epochs = 1000
agent = Agent(gamma=params['gamma'],
                  initial_eps=initial_epsilon,
                  final_eps=final_epsilon,
                  batch_size=batch_size,
                  n_actions=n_actions,
                  input_dims=input_dims,
                  lr=lr)
agent.load_model(model_path)
for step in range(num_epochs):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.choose_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        rewards += reward
        print(reward)
        frame = env.render()
        img.set_data(frame)
        action_text.set_text(f'Step: {actions[action]}')

        fig.canvas.draw()
        fig.canvas.flush_events()
        done = terminated or truncated
        obs = next_obs

plt.ioff()  # Turn off interactive mode
# plt.show()  # Keep the window open after the animation finishes
plt.close()
env.close()
