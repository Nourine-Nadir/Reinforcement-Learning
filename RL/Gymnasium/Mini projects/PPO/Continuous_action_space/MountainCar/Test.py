import json
import torch
import numpy as np
from tqdm import tqdm
from agent import Agent
import gymnasium as gym
from utils import plotLearning


def test(environement=None):
    # Print the device being used
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_device}")

    # ------ HYPERPARAMETERS -------
    with open('params.json', 'r') as f:
        params = json.load(f)["parameters"]

    _nb_episodes, _epochs, _gamma, _gae_lambda, _alpha_clip, _batch_size, \
        _lr, _min_lr, _model_path, _layer1_nodes, _layer2_nodes, _layer3_nodes \
        = \
        (params[key] for key in
         list(params.keys())
         )

    env = environement
    _action_space = env.action_space

    # ------ AGENT -------
    _input_shape = env.observation_space.shape
    agent = Agent(
        input_shape=_input_shape,
        fc1_dims=_layer1_nodes,
        fc2_dims=_layer2_nodes,
        action_space=_action_space,
        epochs=_epochs,
        lr=_lr,
        min_lr=_min_lr,
        gamma=_gamma,
        gae_lambda=_gae_lambda,
        alpha_clip=_alpha_clip,
        batch_size=_batch_size
    )
    agent.load_model(_model_path)
    scores, alpha_history = [], []
    for i in tqdm(range(_nb_episodes)):
        score = 0
        done = False
        obs, info = env.reset()

        while not done:
            action,  _ ,_ = agent.choose_action(obs)
            # print(action)
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            score += reward
            obs = obs_

            env.render()
        scores.append(score)

        avg_score = np.mean(scores[-10:])
        print('Episode', i, 'score %.1f avg score %.1f  ' % (score, avg_score))
