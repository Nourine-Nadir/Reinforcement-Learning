from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
import numpy as np
class CustomFrozenLake(FrozenLakeEnv):
    def __init__(self, goal_reward=100, hole_penalty=-50, step_penalty=-1 ,stuck_penalty=-1, **kwargs):
        super().__init__(**kwargs)
        self.goal_reward = goal_reward
        self.hole_penalty = hole_penalty
        self.step_penalty = step_penalty
        self.stuck_penalty = stuck_penalty

    def step(self, action):
        prev_state = self.unwrapped.s

        state, reward, terminated, truncated, info = super().step(action)

        current_tile = self.desc[self.unwrapped.s // self.ncol, self.unwrapped.s % self.ncol]

        if current_tile in b'H':
            reward = self.hole_penalty  # Apply penalty for falling into a hole
        elif current_tile in b'G':
            reward = self.goal_reward  # Apply higher reward for reaching the goal

        elif prev_state == state :
            reward = self.stuck_penalty  # Apply small penalty for walking on frozen tiles
        else:
            reward = self.step_penalty


        return state, reward, terminated, truncated, info
    def set_goalReward(self, reward):
        self.goal_reward = reward
        return
    def set_stepPenalty(self, penalty):
        self.step_penalty = penalty
        return
    def set_stuckPenalty(self, penalty):
        self.stuck_penalty = penalty
        return

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.s = self.get_random_initial_state()
        return self.s, {}

    def get_random_initial_state(self):
        valid_states = [i for i in range(64)]
        return np.random.choice(valid_states)