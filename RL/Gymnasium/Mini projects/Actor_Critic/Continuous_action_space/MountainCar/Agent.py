import os, pickle
import torch as T
import torch.optim as optim
from models import GenericNetwork
from torch.distributions import Normal

class Agent():
    def __init__(self,
                 input_shape,
                 fc1_dims: int,
                 fc2_dims: int,
                 action_space,
                 n_outputs: int,
                 actor_lr: float = 5e-5,
                 critic_lr: float = 1e-4
                 ):
        # Initialize attributes
        self.action_space = action_space
        self.n_actions = action_space.shape[0]
        self.n_outputs = n_outputs
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_shape = input_shape
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.log_probs = None

        # Actor & Critic saved_models
        self.Actor = GenericNetwork(lr=self.actor_lr,
                                    min_lr=1e-5,
                                    input_shape=self.input_shape,
                                    fc1_dims=self.fc1_dims,
                                    fc2_dims=self.fc2_dims,
                                    n_actions=(self.n_actions*2),
                                    action_space= self.action_space)
        self.Critic = GenericNetwork(lr=self.critic_lr,
                                     min_lr=1e-5,
                                     input_shape=self.input_shape,
                                     fc1_dims=self.fc1_dims,
                                     fc2_dims=self.fc2_dims,
                                     n_actions= 1,
                                     action_space= self.action_space)

    # Sample action from Actor network
    def choose_action(self, obs):
        action, self.log_probs = self.Actor.sample(obs, self.n_outputs)

        return action.item()

    # Run gradient on both Actor and Critic networks
    def update_parameters(self, state, reward, next_state, done):
        self.Actor.optimizer.zero_grad()
        self.Critic.optimizer.zero_grad()

        state_value = self.Critic.forward(state)
        state_value_ = self.Critic.forward(next_state)

        reward = T.tensor(reward, dtype=T.float32, device=self.Actor.device)

        advantage = (reward + 0.99 * state_value_ * (1-int(done))) - state_value

        actor_loss = -self.log_probs * advantage
        critic_loss = T.square(advantage)

        total_loss = actor_loss + critic_loss
        total_loss.backward()
        print(f'actor_loss: {actor_loss}, critic_loss: {critic_loss}')

        self.Actor.optimizer.step()
        self.Critic.optimizer.step()

        self.Actor.lr_decay()
        self.Critic.lr_decay()


    def get_lr(self):
        return self.Actor.get_lr(), self.Critic.get_lr()

    def save_model(self, PATH):
        os.makedirs(os.path.dirname(PATH), exist_ok=True)
        T.save(self.Actor.state_dict(), PATH)

    def load_model(self, PATH ):
        try:
            print('imported model')
            self.Actor.load_state_dict(T.load(PATH, weights_only=True))
        except FileNotFoundError:
            print(f"Error: Could not find model files at {PATH }")
        except RuntimeError as e:
            print(f"Error loading model: {e}")