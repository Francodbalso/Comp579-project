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
    def __init__(self, state_dim, h_dim, action_dim, action_space):
        self.action_dim = action_dim
        # store action bounds
        self.upper_a_bounds = torch.from_numpy(action_space.high).to(torch.float32)
        self.lower_a_bounds = torch.from_numpy(action_space.low).to(torch.float32)
        
        # make an mlp with last layer's weights extra small initially to help tanh gradients
        self.mlp = MLP(state_dim, h_dim, 2 * action_dim)
        self.mlp.layers[-1].weight.data *= 0.01
        self.mlp.layers[-1].bias.data *= 0.01
    
    def get_means_logstds(self, s):
        # expecting s to be of shape (state_dim)
        # first half of outputs will be the means, other half will be logstds
        outputs = self.mlp(s)
        means = outputs[:self.action_dim]
        # bound the log stds, using tanh, to avoid numerical instabilities later on
        lowerbound, upperbound = -10, 2
        logstds = outputs[self.action_dim:]
        logstds = lowerbound + 0.5 * (upperbound - lowerbound) * (torch.tanh(logstds) + 1)
        return means, logstds

    def get_action_logprob(self, s):
        # expecting s to be of shape (state_dim)
        means, logstds = self.get_means_logstds(s)
        normal = torch.distributions.Normal(means, torch.exp(logstds))
        raw_action = normal.rsample()  # uses the reparametrization trick to allow differentiability
        squashed_action = torch.tanh(raw_action)
        scale = 0.5 * (self.upper_a_bounds-self.lower_a_bounds)
        bias = scale + self.lower_a_bounds
        scaled_action = scale * squashed_action + bias

        # now need to compute log probs of scaled actions taking into account the tanh + affine transformation
        raw_logprob = normal.log_prob(raw_action).sum(dim=1)
        log_det_jacobian = torch.log(scale * (1 - squashed_action ** 2) + 1e-6).sum(dim=1)
        log_prob = raw_logprob - log_det_jacobian

        return scaled_action, log_prob
