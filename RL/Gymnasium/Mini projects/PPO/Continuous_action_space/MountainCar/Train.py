import json
import torch
import numpy as np
from tqdm import tqdm
from agent import Agent
from utils import plotLearning


def train( plot_name, save_model=False, environement =None):
    # Print the device being used
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_device}")

    # ------ HYPERPARAMETERS -------
    with open('params.json', 'r') as f:
        params = json.load(f)["parameters"]

    _nb_episodes, _epochs, _gamma, _gae_lambda, _alpha_clip, _batch_size,  \
        _lr,_min_lr, _model_path,_layer1_nodes, _layer2_nodes, _entropy_factor \
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
                  lr= _lr,
                  min_lr= _min_lr,
                  gamma=_gamma,
                  gae_lambda=_gae_lambda,
                  alpha_clip=_alpha_clip,
                  batch_size=_batch_size,
                  entropy_factor=_entropy_factor
                  )

    # ------ TRAINING -------
    scores, alpha_history = [], []
    total_steps, learn_iters = 0, 0
    for i in tqdm(range(_nb_episodes)):
        score = 0

        done = False
        obs, info = env.reset()
        ep_steps = 0
        while not done:
            action, log_prob, value = agent.choose_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            agent.store_transition(obs, action, log_prob, value, reward, float(done))
            # print('action stored ! ')
            if total_steps % (_batch_size*4) == 0 and total_steps > 0:
                # print('Update')
                agent.update_parameters()
                learn_iters += 1

            score += reward
            obs = obs_
            ep_steps+=1
            total_steps+=1

        # agent.update_parameters()
        # learn_iters +=1


        scores.append(score)

        # ------ PRINTING -------
        if i % 1 == 0:  # Print less frequently
            avg_score = np.mean(scores[-5:])
            actor_lr, critic_lr = agent.get_lr()

            # print('actor lr %.5f critic lr %.5f ' % (actor_lr, critic_lr))
            print('Episode', i, 'score %.1f avg score %.1f  ep_steps %.f total_steps %.f ' % (score, avg_score, ep_steps, total_steps))
            print(f' lr {actor_lr} ')
            # if score >0 :
            #     agent.save_model(_model_path)
            #
            #     x = [i + 1 for i in range(_nb_episodes)]
            #     filename = plot_name + str(_epochs) + ' epochs' + ' lr ' + str(_lr) + \
            #                ' min lr ' + str(_min_lr) + '.png'
            #
            #     plotLearning(x, scores, filename=filename)
            #     break

        # ------ Model saving -------
    if save_model:
        agent.save_model(_model_path)

        x = [i + 1 for i in range(_nb_episodes)]
        filename = plot_name+str(_epochs) + ' epochs'+ ' lr ' +str( _lr)+\
                   ' min lr ' +str(_min_lr) + '.png'

        plotLearning(x, scores, filename=filename)
