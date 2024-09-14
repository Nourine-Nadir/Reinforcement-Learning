import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym


def plotLearning(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.scatter(x, running_avg, color="C1")
    # ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    # ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    # ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    # ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=5):
        super(SkipEnv, self).__init__(env)  # not necessary in python 3 to pass the class and self to super method
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = truncated or terminated
            if done:
                break
        return obs, total_reward, terminated, truncated, info


class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(80, 80, 1), dtype=np.uint8)

    # This method is called every time an observation is returned from the environment.
    def observation(self, obs):
        return PreProcessFrame.process(obs)

    @staticmethod
    # we use a static method because we are not accessing any class variable, but just the ones passed to this method
    def process(frame):
        new_frame = frame.astype(np.float32)
        new_frame = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
        new_frame = new_frame[35:195:2, ::2].reshape(80, 80, 1)

        return new_frame.astype(np.uint8)

class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,  # expecting normalized input
                                                shape=(self.old_shape[-1],
                                                       self.old_shape[0],self.old_shape[1]),
                                                dtype=np.float32)
    def observation(self, observation):
        # move channel to first dimension
        return np.moveaxis(observation, 2, 0)


class ScaleFrame(gym.ObservationWrapper):
    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super().__init__(env)
        self.n_steps = n_steps
        self.observation_space = gym.spaces.Box(low= env.observation_space.low.repeat(n_steps, axis=0),
                                                high = env.observation_space.high.repeat(n_steps, axis=0),
                                                dtype=np.float32)
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        for i in range(self.n_steps):
            self.buffer[i] = observation
        return self.observation(observation), info

    def observation(self, observation):

        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer

def make_env(env_name, render_mode=None):
    env = gym.make(env_name, render_mode=render_mode)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MoveImgChannel(env)  # It is crucial to move the image channel to the 0th dim before the BufferWrapepr so it can stack images
    env = BufferWrapper(env, 5)
    return ScaleFrame(env)
