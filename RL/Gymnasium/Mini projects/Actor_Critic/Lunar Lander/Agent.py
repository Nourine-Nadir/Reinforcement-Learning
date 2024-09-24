import os, pickle
import torch as T
import torch.optim as optim
from Actor_Critic import ActorCritic


class Agent():
    def __init__(self,
                 batch_size: float,
                 input_shape,
                 layer1_nodes,
                 layer2_nodes,
                 layer3_nodes,
                 n_actions: int,
                 max_mem_size: int = 10_000,
                 gamma: float = 0.99,
                 initial_eps: float = 1.0,
                 final_eps: float = 0.01,
                 eps_decay: float = 1e-4,
                 lr: float = 1e-4,
                 update_freq : int =100
                 ):
        self.gamma = gamma
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.eps_decay = eps_decay
        self.lr = lr
        self.eps = initial_eps
        self.eps_decay = eps_decay
        self.final_eps = final_eps
        self.mem_counter = 0
        self.sequence_length = 32  # You can adjust this value

        self.target_update_frequency = update_freq
        self.learn_step_counter = 0
        self.ActorCritic = ActorCritic()
        self.optimizer = optim.Adam(self.ActorCritic.parameters(), lr=self.lr, betas=(0.9, 0.999))






    def learn(self):

        self.optimizer.zero_grad()
        loss = self.ActorCritic.calculateLoss()
        loss.backward()
        self.optimizer.step()
        self.ActorCritic.clearMemory()


        self.learn_step_counter += 1
        self.eps = max((self.eps - self.eps_decay), self.final_eps)



    def save_model(self, PATH):
        T.save(self.Actor.state_dict(), PATH)

    def load_model(self, q_eval_path ):
        try:
            self.Actor = T.load(q_eval_path, map_location=self.device)
        except FileNotFoundError:
            print(f"Error: Could not find model files at {q_eval_path }")
        except RuntimeError as e:
            print(f"Error loading model: {e}")


    def save_agent(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f)