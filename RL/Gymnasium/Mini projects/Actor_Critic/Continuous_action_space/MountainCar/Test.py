import json
import torch
import numpy as np
from tqdm import tqdm
from Agent import Agent
import gymnasium as gym
from utils import plotLearning


def test(environement=None):
    # Print the device being used
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_device}")

    with open('params.json', 'r') as f:
        params = json.load(f)["parameters"]

    _nb_episodes, _epochs, _gamma, _alpha, _final_alpha, _initial_eps, _eps_decay, \
        _final_eps, _batch_size, \
        _lr, _min_lr, _max_memory_size, _model_path, _agent_path, \
        _layer1_nodes, _layer2_nodes, _layer3_nodes, _update_freq \
        = \
        (params[key] for key in
         list(params.keys())
         )

    env = environement
    _n_actions = env.action_space.shape[0]
    print('action space : ', env.action_space.shape)
    print('obs space : ', env.observation_space.shape)

    _input_shape = [env.observation_space.shape]
    agent = Agent(
        input_shape=_input_shape,
        fc1_dims=_layer1_nodes,
        fc2_dims=_layer2_nodes,
        n_actions=_n_actions,
        n_outputs=1,
        actor_lr=0.00005,
        critic_lr=0.0001
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

            score += reward
            obs = obs_

            env.render()
        scores.append(score)

        avg_score = np.mean(scores[-10:])
        print('Episode', i, 'score %.1f avg score %.1f  ' % (score, avg_score))
