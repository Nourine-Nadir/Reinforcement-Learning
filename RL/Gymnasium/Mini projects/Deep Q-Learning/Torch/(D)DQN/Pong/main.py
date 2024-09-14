import numpy as np
from Agent import Agent
from utils import make_env, plotLearning
import json
from tqdm import tqdm
import timeit

with open('params.json', 'r') as f:
    params = json.load(f)["parameters"]

num_games, gamma, initial_eps, eps_decay, \
    final_eps, batch_size, n_actions, input_dims, \
    lr, max_memory_size, model_path, Q_eval_path, Q_target_path, agent_path, \
    layer1_nodes, layer2_nodes, layer3_nodes, update_freq \
    = \
    (params[key] for key in
     list(params.keys())
     )
input_dims = (input_dims['nb_images'], input_dims['height'], input_dims['width'])

if __name__ == '__main__':
    env = make_env(env_name='PongNoFrameskip-v4')
    load_checkpoint = False
    best_score = -21
    agent = Agent(update_freq=update_freq,
                  input_dims=input_dims,
                  layer1_nodes=layer1_nodes,
                  n_actions=n_actions,
                  lr=lr,
                  epsilon=initial_eps,
                  eps_decay=eps_decay,
                  eps_final=final_eps,
                  gamma=gamma,
                  mem_size=max_memory_size,
                  batch_size=batch_size,
                  q_eval_filename=Q_eval_path,
                  q_target_filename=Q_target_path)

    if load_checkpoint:
        agent.load_models(Q_eval_path, Q_target_path)

    scores, eps_history = [], []
    n_steps = 0

    for i in tqdm(range(num_games)):

        score = 0
        observation, info = env.reset()
        done = False

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            #             print(observation.shape)
            #             print(observation_.shape)
            score += reward
            n_steps += 1
            if not load_checkpoint:
                agent.store_transition(observation, action, reward,
                                       observation_, int(done))
                agent.learn()
                # agent.update_epsilon()
            else:
                env.render()
            observation = observation_

        scores.append(score)
        print('score:', scores)
        avg_score = np.mean(scores[-100:])
        print('episodes', i, 'score', score, 'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            print('avg score %.2f better than the best score %.2f' % (avg_score, best_score))
            agent.save_models(Q_eval_path, Q_target_path)
            best_score = avg_score

        eps_history.append(agent.epsilon)
    x = [i + 1 for i in range(num_games)]
    filename = 'PongNoFrameskip-v4.png'
    plotLearning(x, scores, eps_history, filename)
