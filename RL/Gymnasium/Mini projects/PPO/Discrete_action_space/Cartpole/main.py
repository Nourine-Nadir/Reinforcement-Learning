import gymnasium as gym
from Train import train
from Test import test

if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human')


    # train(save_model=True, plot_name='Cartpole PPO ', environement=env)
    #
    test(env)