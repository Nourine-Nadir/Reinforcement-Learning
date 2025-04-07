import gymnasium as gym
from Train import train
from Test import test

if __name__ == '__main__':
    env = gym.make('MountainCarContinuous-v0', render_mode='human', max_episode_steps=8000)
    env.metadata['render_fps'] = 60  # Adjust FPS

    train(save_model=True, plot_name='Mountain PPO ', environement=env)
    #
    # test(env)