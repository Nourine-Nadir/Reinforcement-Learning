import numpy as np
from Agent import Agent
from utils import make_env, plotLearning
import json
from tqdm import tqdm
with open('params.json', 'r') as f:
    params = json.load(f)["parameters"]

num_games, gamma, initial_eps, eps_decay, \
    final_eps, batch_size, n_actions, input_dims, \
    lr, max_memory_size, model_path, Q_eval_path, Q_target_path, agent_path, \
    layer1_dims, layer2_dims, layer3_dims, update_freq \
    = \
    (params[key] for key in
     list(params.keys())
     )

if __name__ == '__main__':
    env = make_env(env_name='PongNoFrameskip-v4')
    load_checkpoint = False
    best_score = -21

    agent = Agent()

    if load_checkpoint:
        agent.load_models()


    scores, eps_history = [], []
    n_steps = 0

    for i in tqdm(range(num_games)):
        score = 0
        observation = env.reset()
        done = False
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            n_steps += 1
            if not load_checkpoint:
                agent.store_transition(observation, action, reward,
                                       observation_, int(done))
                agent.learn()
            else:
                env.render()
            observation = observation_

        scores.append(score)
        print('score:', scores)
        avg_score = np.mean(scores[-100:])
        print('episodes', i, 'score', score, 'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon , 'steps', n_steps)

        if avg_score > best_score:
            agent.save_models()
            print('avg score %.2f better than the best score %.2f' % (avg_score, best_score))
            best_score = avg_score

        eps_history.append(agent.epsilon)
    x = [i + 1 for i in range(num_games)]
    filename = 'PongNoFrameskip-v4.png'
    plotLearning(x, scores, eps_history, filename)