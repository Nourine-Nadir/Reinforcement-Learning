import numpy as np
from Agent import Agent
from utils import make_env, plotLearning
import json
from tqdm import tqdm
import gymnasium as gym
import torch

# Print the device being used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open('params.json', 'r') as f:
    params = json.load(f)["parameters"]

nb_episodes, gamma, initial_eps, eps_decay, \
    final_eps, batch_size, n_actions, input_shape, \
    lr, max_memory_size, model_path, Q_eval_path, Q_target_path, agent_path, \
    layer1_nodes, layer2_nodes, layer3_nodes, update_freq \
    = \
    (params[key] for key in
     list(params.keys())
     )

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
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

    scores, eps_history = [], []
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
        except RuntimeError as e:
            print(f"RuntimeError in learning step: {e}")
            print(f"Current observation shape: {obs.shape}")
            print(f"Current action: {action}")
            raise e
        scores.append(score)
        eps_history.append(agent.eps)

        if i % 100 == 0:  # Print less frequently
            avg_score = np.mean(scores[-100:])
            print('Episode', i, 'score %.1f avg score %.1f epsilon %.3f' % (score, avg_score, agent.eps))

    agent.save_agent(agent_path)
    x = [i + 1 for i in range(nb_episodes)]
    filename = 'lunar_lander.png'
    plotLearning(x, scores, eps_history, filename)