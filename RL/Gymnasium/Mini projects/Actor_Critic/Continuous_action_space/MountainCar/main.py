import gymnasium as gym
from Train import train
from Test import test

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0')


    train(save_model=True, plot_name='MontainCar SAC without log_probs scaling', environement=env)
    # test(env)