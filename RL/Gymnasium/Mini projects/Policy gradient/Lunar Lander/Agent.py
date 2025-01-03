import os, pickle
import torch as T
import torch.optim as optim
from Policy_gradient import PolicyGradient


class Agent():
    def __init__(self,
                 batch_size: float,
                 input_shape,
                 layer1_nodes,
                 layer2_nodes,
                 layer3_nodes,
                 n_actions: int,
                 gamma: float = 0.99,
                 alpha: float = 0.99,
                 final_alpha = 0.01,
                 initial_eps: float = 1.0,
                 final_eps: float = 0.01,
                 eps_decay: float = 1e-4,
                 lr: float = 1e-4,
                 min_lr: float = 1e-3,
                 ):
        self.gamma = gamma
        self.alpha = alpha
        self.final_alpha = final_alpha
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.eps_decay = eps_decay
        self.lr = lr
        self.min_lr = min_lr
        self.eps = initial_eps
        self.eps_decay = eps_decay
        self.final_eps = final_eps

        self.policy = PolicyGradient(alpha=self.alpha, final_alpha=self.final_alpha)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.__scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch : max(0.99 ** epoch, (self.min_lr/self.lr) )
        )
        print('lr ,',self.lr , ' min_lr ',self.min_lr )
        print('dividing ',self.min_lr / self.lr )




    def learn(self):

        self.optimizer.zero_grad()
        loss = self.policy.calculateLoss()
        loss.backward()
        self.optimizer.step()
        self.policy.clearMemory()



    def lr_decay(self):
        self.__scheduler.step()

    def get_lr(self):
        return self.__scheduler.get_last_lr()[0]
    def save_model(self, PATH):
        T.save(self.policy.state_dict(), PATH)

    def load_model(self, PATH ):
        try:
            print('imported model')
            self.policy.load_state_dict(T.load(PATH, weights_only=True))
        except FileNotFoundError:
            print(f"Error: Could not find model files at {PATH }")
        except RuntimeError as e:
            print(f"Error loading model: {e}")


    def save_agent(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(path, 'wb') as f:
            pickle.dump(self, f)