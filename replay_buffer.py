import numpy as np
class ReplayBuffer():
    """
    FIFO replay buffer which uses random sampling.
    Implemented as a np array
    """
    def __init__(self, max_len, obs_dim, action_dim):
        self.max_len = max_len
        #self.arr = np.zeros((max_len, 2*obs_dim+1+1+1))
        self.arr = np.zeros((max_len, 2*obs_dim+action_dim+1+1))
        self.cur_ind = 0
        self.is_full = False
        self.obs_dim = obs_dim
        self.action_dim = action_dim
    def add_entry(self, state, next_state, action, reward, done):
    #def add_entry(self, state, next_state, reward):
        #add an entry ot the array  (consists of adding a single vector of state, next state, action,)
        self.arr[self.cur_ind, :]=np.concatenate((state, next_state, action, np.array([reward], dtype=float), np.array([done], dtype=float)))
        #self.arr[self.cur_ind, :]=np.concatenate((state, next_state, np.array([reward], dtype=float)))

        self.cur_ind = (self.cur_ind+1)
        if self.cur_ind >= self.max_len:
            self.is_full = True
            self.cur_ind = 0

    def sample(self, num_samples):
        if self.is_full:
            max_ind = self.max_len
        else:
            max_ind = self.cur_ind
        idx = np.random.choice(range(max_ind), num_samples, replace=False)
        #print(idx)
        all_vals = self.arr[idx,:]
        states = all_vals[:, :self.obs_dim]
        next_states = all_vals[:, self.obs_dim:2*self.obs_dim]
        actions = all_vals[:, 2*self.obs_dim:2*self.obs_dim+self.action_dim]
        rewards = all_vals[:, -2:-1]
        dones = all_vals[:, -1:]
        #print(states)
        #print(rewards)
        return states, next_states, actions, rewards, dones
        #return states, next_states, rewards, dones

