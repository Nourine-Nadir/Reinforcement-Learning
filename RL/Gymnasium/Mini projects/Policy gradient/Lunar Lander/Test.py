import json
import torch
import numpy as np
from tqdm import tqdm
from Agent import Agent
import gymnasium as gym
from utils import plotLearning


def test():
    # Print the device being used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open('params.json', 'r') as f:
        params = json.load(f)["parameters"]

    nb_episodes, gamma, alpha, final_alpha, initial_eps, eps_decay, \
        final_eps, batch_size, n_actions, input_shape, \
        lr, min_lr, max_memory_size, model_path, agent_path, \
        layer1_nodes, layer2_nodes, layer3_nodes, update_freq \
        = \
        (params[key] for key in
         list(params.keys())
         )

    env = gym.make('LunarLander-v2', render_mode='human',max_episode_steps=600)
    agent = Agent(layer1_nodes=layer1_nodes,
                  layer2_nodes=layer2_nodes,
                  layer3_nodes=layer3_nodes,
                  gamma=gamma,
                  initial_eps=initial_eps,
                  eps_decay=eps_decay,
                  final_eps=final_eps,
                  batch_size=batch_size,
                  n_actions=4,
                  input_shape=input_shape,
                  lr=lr,
                  update_freq=update_freq,
                  max_mem_size=max_memory_size)
    agent.load_model(model_path)

    scores, alpha_history = [], []
    for i in tqdm(range(nb_episodes)):
        score = 0
        done = False
        obs, info = env.reset()

        while not done:
            action = agent.ActorCritic(obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            score += reward
            agent.ActorCritic.rewards.append(reward)


            obs = obs_
            done = terminated or truncated

            env.render()
        scores.append(score)
        alpha_history.append(agent.ActorCritic.alpha.item())

        avg_score = np.mean(scores[-10:])
        print('Episode', i, 'score %.1f avg score %.1f  alpha %.3f' % (score, avg_score, agent.ActorCritic.alpha))
