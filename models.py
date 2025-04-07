import torch
import torch.nn as nn

class MLP(nn.Module):
    '''basic ReLU activated 2 layer mlp with no output activation'''
    def __init__(self, in_dim, h_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_dim, h_dim),
                                    nn.ReLU(),
                                    nn.Linear(h_dim, h_dim),
                                    nn.ReLU(),
                                    nn.Linear(h_dim, out_dim))
    
    def forward(self, x):
        return self.layers(x)
    

class Qu():
    '''
    The option value model Q_U(s, w, a) from the paper, except every instance
    of this class will correspond to a specific option w.
    in_dim should be state_dim + action_dim
    '''
    def __init__(self, s_a_dim, h_dim):
        self.mlp = MLP(s_a_dim, h_dim, 1)
    
    def get_value(self, s_a):
        # assume already normalized and concatenated state and action
        return self.mlp(s_a)
    

class TerminationFunction():
    '''
    The termination model which takes in a state and returns a probability of termination.
    Again each instance of this class will correspond to a specific option.
    in_dim should be the state dimension.
    '''
    def __init__(self, state_dim, h_dim):
        self.mlp = MLP(state_dim, h_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def get_term_prob(self, s):
        # assume already normalized state
        return self.sigmoid(self.mlp(s))


class IntraOptionPolicy():
    '''
    Stochastic policy, again one instance per option.
    The network outputs means and log standard deviations for each action
    in the form of a 1D array (means, log_stds).
    '''
    def __init__(self, state_dim, h_dim, action_dim):
        self.action_dim = action_dim
        self.mlp = MLP(state_dim, h_dim, 2 * action_dim)
    
    def get_means_logstds(self, s):
        # expecting s to be of shape (batch, state_dim)
        # first half of outputs will be the means, other half will be logstds
        outputs = self.mlp(s)
        means = outputs[:, :self.action_dim]
        logstds = outputs[:, self.action_dim:]
        return means, logstds