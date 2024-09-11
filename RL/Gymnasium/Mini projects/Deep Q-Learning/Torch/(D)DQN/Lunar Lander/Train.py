import gymnasium as gym
import numpy as np
from utils import plotLearning
from DQN_q_twoNetworks import Agent
import json
from tqdm import tqdm

with open('params.json', 'r') as f:
    params = json.load(f)["parameters"]

nb_episodes, gamma, initial_eps, eps_decay, \
    final_eps, batch_size , n_actions, input_dims, \
    lr,max_memory_size,model_path,agent_path, \
    layer1_dims, layer2_dims, layer3_dims, \
   = \
    (params[key] for key in
        list(params.keys())
    )

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=gamma,
                  initial_eps=initial_eps,
                  eps_decay=eps_decay,
                  final_eps=final_eps,
                  batch_size=batch_size,
                  n_actions=n_actions,
                  input_dims=input_dims,
                  lr=lr)
    scores, eps_history = [], []
    j = nb_episodes
    for i in tqdm(range(nb_episodes)):
        score = 0
        done = False
        obs, info = env.reset()

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

    agent.save_agent(agent_path)
    x = [i + 1 for i in range(j)]
    filename = 'lunar_lander.png'
    plotLearning(x, scores, eps_history, filename)

