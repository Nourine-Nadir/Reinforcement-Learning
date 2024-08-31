from Custom_Env import CustomFrozenLake
from Q_value_Agent import Agent
import pickle,json
from tqdm import tqdm
import gymnasium as gym

with open('params.json', 'r') as f:
    params = json.load(f)['parameters']

env = CustomFrozenLake(map_name="8x8", is_slippery=False, render_mode='rgb_array')

learning_rate = params['learning_rate']
n_episodes = params['n_episodes']
initial_epsilon = params['initial_eps']
epsilon_decay = params['eps_decay']
final_epsilon = params['final_eps']
agent_path = params['agent_path']

agent = Agent(
    learning_rate = learning_rate,
    initial_epsilon = initial_epsilon,
    epsilon_decay = epsilon_decay,
    final_epsilon = final_epsilon,
    env = env
)
#%%
env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
env = gym.wrappers.TimeLimit(env, max_episode_steps=60)

rewards = 0
if __name__ == '__main__':
    for episode in tqdm(range(n_episodes)):

        obs, info = env.reset()
        done = False

        # play one episode
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            rewards += reward
            # update the agent
            agent.update_q_values(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

    agent.save_agent(agent_path)
