import json
import torch
import numpy as np
from tqdm import tqdm
from Agent import Agent
import gymnasium as gym
from utils import plotLearning


def train( plot_name, save_model=False):
    # Print the device being used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open('params.json', 'r') as f:
        params = json.load(f)["parameters"]

    nb_episodes, gamma,alpha, final_alpha, initial_eps, eps_decay, \
        final_eps, batch_size, n_actions, input_shape, \
        lr,min_lr, max_memory_size, model_path, agent_path, \
        layer1_nodes, layer2_nodes, layer3_nodes, update_freq \
        = \
        (params[key] for key in
         list(params.keys())
         )

    env = gym.make('LunarLander-v2')
    agent = Agent(layer1_nodes=layer1_nodes,
                  layer2_nodes=layer2_nodes,
                  layer3_nodes=layer3_nodes,
                  gamma=gamma,
                  alpha=alpha,
                  final_alpha=final_alpha,
                  initial_eps=initial_eps,
                  eps_decay=eps_decay,
                  final_eps=final_eps,
                  batch_size=batch_size,
                  n_actions=4,
                  input_shape=input_shape,
                  lr=lr,
                  min_lr=min_lr,
                  update_freq=update_freq,
                  max_mem_size=max_memory_size)

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

        try:
            agent.learn()
            if i % 10 == 0 :
                agent.scheduler.step()
                # print(agent.scheduler.get_lr())

            # print(agent.ActorCritic.alpha.item())
        except RuntimeError as e:
            print(f"RuntimeError in learning step: {e}")
            print(f"Current observation shape: {obs.shape}")
            print(f"Current action: {action}")
            raise e
        scores.append(score)
        alpha_history.append(agent.ActorCritic.alpha.item())

        if i % 100 == 0:  # Print less frequently
            avg_score = np.mean(scores[-100:])
            print('Episode', i, 'score %.1f avg score %.1f   alpha %.3f lr %.5f' % (score, avg_score, agent.ActorCritic.alpha, agent.scheduler.get_lr()[0]))
    if save_model:
        # agent.save_agent(agent_path)
        agent.save_model(model_path)
        x = [i + 1 for i in range(nb_episodes)]
        filename = plot_name+' lr '+ str(lr)+' min_lr '+ str(min_lr) +'.png'
        plotLearning(x, scores, alpha_history, filename)