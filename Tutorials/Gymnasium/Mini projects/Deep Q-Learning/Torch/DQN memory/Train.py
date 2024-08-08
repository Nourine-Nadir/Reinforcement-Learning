import gymnasium as gym
import numpy as np
from utils import plotLearning
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

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    print(env.observation_space.shape[0])
    agent = Agent(gamma=gamma,
                  initial_eps=initial_epsilon,
                  final_eps=final_epsilon,
                  batch_size=batch_size,
                  n_actions=n_actions,
                  input_dims=input_dims,
                  lr=lr)
    scores, eps_history = [], []
    n_episodes = 1000
    j = n_episodes
    avg_score = -9999
    for i in range(n_episodes):
        score = 0
        done = False
        obs, info = env.reset()
        if avg_score > -60:
            j = i
            break
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            score += reward
            agent.store_transition(obs, action, reward, obs_, done)
            agent.learn()
            obs = obs_
            done = terminated or truncated
        scores.append(score)
        eps_history.append(agent.eps)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print('Episode', i, 'score %.1f avg score %.1f epsilon %.3f' % (score, avg_score, agent.eps))

    agent.save_model(model_path)
    x = [i + 1 for i in range(j)]
    filename = 'lunar_lander.png'
    plotLearning(x, scores, eps_history, filename)

