import json

import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm import tqdm
from Agent import Agent
import gymnasium as gym
from utils import plotLearning


def train( plot_name, save_model=False, environement =None):
    # Print the device being used
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_device}")

    with open('params.json', 'r') as f:
        params = json.load(f)["parameters"]

    _nb_episodes, _epochs, _gamma, _alpha, _final_alpha, _initial_eps, _eps_decay, \
        _final_eps, _batch_size,  \
        _actor_lr, _critic_lr, _max_memory_size, _model_path, _agent_path, \
        _layer1_nodes, _layer2_nodes, _layer3_nodes, _update_freq \
        = \
        (params[key] for key in
         list(params.keys())
         )

    env = environement
    _action_space = env.action_space
    print('action space : ', env.action_space)
    print('obs space : ', env.observation_space.shape)

    _input_shape = [env.observation_space.shape]
    agent = Agent(
                  input_shape=_input_shape,
                  fc1_dims=_layer1_nodes,
                  fc2_dims=_layer2_nodes,
                  action_space=_action_space,
                  n_outputs=1,
                  actor_lr= _actor_lr,
                  critic_lr= _critic_lr
                  )

    scores, alpha_history = [], []
    for i in tqdm(range(_nb_episodes)):
        score = 0
        done = False
        obs, info = env.reset()

        while not done:
            action = [agent.choose_action(obs)]
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.update_parameters(obs, reward, obs_, done)

            score += reward
            obs = obs_


        scores.append(score)

        if i % 2 == 0:  # Print less frequently
            avg_score = np.mean(scores[-10:])
            actor_lr, critic_lr = agent.get_lr()
            print('actor lr %.5f critic lr %.5f ' % (actor_lr, critic_lr))
            print('Episode', i, 'score %.1f avg score %.1f   ' % (score, avg_score))
        # print(f'Episode {i} score {score}')
    if save_model:
        # agent.save_agent(agent_path)
        # agent.save_model(_model_path)
        x = [i + 1 for i in range(_nb_episodes)]
        filename = plot_name+' actor lr ' +str(_actor_lr)+\
                   ' critic lr ' +str(_critic_lr) + '.png'
        plotLearning(x, scores, filename=filename)