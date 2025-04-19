import numpy as np

class ReplayBuffer():
    """
    Replay buffer which uses random sampling.
    Implemented as a np array
    """
    def __init__(self, max_len, obs_dim, action_dim):
        self.max_len = max_len
        self.arr = np.zeros((max_len, 2*obs_dim+action_dim+1+1+1+1+1+1))
        self.cur_ind = 0
        self.is_full = False
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def add_entry(self, state, next_state, action, v, v_next, max_v_next, logprob, reward, done):
        #add an entry to the array  (consists of adding a single vector of state, next state, action, reward, done)
        if not self.is_full:
            self.arr[self.cur_ind, :] = np.concatenate((state, 
                                                        next_state, 
                                                        action,
                                                        np.array([v], dtype=float),
                                                        np.array([v_next], dtype=float),
                                                        np.array([max_v_next], dtype=float),
                                                        np.array([logprob], dtype=float),
                                                        np.array([reward], dtype=float), 
                                                        np.array([done], dtype=float)))
            self.cur_ind += 1
            if self.cur_ind >= self.max_len:
                self.is_full = True

    def sample(self, num_samples):
        max_ind = self.cur_ind
        if num_samples >= max_ind:
            idx = np.arange(max_ind)
        else:
            idx = np.random.choice(range(max_ind), num_samples, replace=False)
        
        all_vals = self.arr[idx,:]
        states = all_vals[:, :self.obs_dim]
        next_states = all_vals[:, self.obs_dim:2*self.obs_dim]
        actions = all_vals[:, 2*self.obs_dim:2*self.obs_dim+self.action_dim]
        vs = all_vals[:, -6]
        next_vs = all_vals[:, -5]
        max_next_vs = all_vals[:, -4]
        logprobs = all_vals[:, -3]
        rewards = all_vals[:, -2]
        dones = all_vals[:, -1]
        
        return states, next_states, actions, vs, next_vs, max_next_vs, logprobs, rewards, dones
    
    def empty(self):
        self.arr = np.zeros((self.max_len, 2*self.obs_dim+self.action_dim+1+1+1+1+1+1))
        self.cur_ind = 0
        self.is_full = False

