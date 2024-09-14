import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memories = np.zeros((2, self.mem_size, *input_shape),# 2 for cuurent and next state memory
                                     dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

    def store_transition(self, state, action, reward, state_, done):
        # print('state',state.shape)
        # print('state',state_.shape)
        index = self.mem_cntr % self.mem_size
        self.state_memories[0][index]= state
        self.state_memories[1][index]= state_
        self.action_memory[index]= action
        self.reward_memory[index]= reward
        self.terminal_memory[index]= done

        self.mem_cntr+=1

    def sample_buffer(self, batch_size):
        # print('sampling')
        max_mem = min(self.mem_size,self.mem_cntr)
        batch = np.random.choice(max_mem,size= batch_size, replace=False)
        batch_index = np.arange(batch_size, dtype=np.int32)

        states = self.state_memories[0][batch]
        states_ = self.state_memories[1][batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminals