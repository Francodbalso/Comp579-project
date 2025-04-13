import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

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
    

class Qw():
    '''
    The option value model Q_U(s, w) from the paper, except every instance
    of this class will correspond to a specific option w.
    in_dim should be state_dim
    '''
    def __init__(self, s_dim, h_dim):
        self.mlp = MLP(s_dim, h_dim, 1)
    
    def get_value(self, s):
        # assume already normalized state
        return self.mlp(s)
    

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
        self.mlp.layers[-1].weight.data *= 0.1
        self.mlp.layers[-1].bias.data *= 0.1
    
    def get_means_logstds(self, s):
        # expecting s to be of shape (state_dim)
        # first half of outputs will be the means, other half will be logstds
        outputs = self.mlp(s)
        means = outputs[:self.action_dim]
        # bound the log stds, using tanh, to avoid numerical instabilities later on
        # can also change the slope of the tanh to avoid needing to make weights small initially
        lowerbound, upperbound = -10, 2
        logstds = outputs[self.action_dim:]
        logstds = lowerbound + 0.5 * (upperbound - lowerbound) * (torch.tanh(logstds) + 1) 
        return means, logstds

    def get_action_logprob_entropy(self, s):
        # expecting s to be of shape (state_dim)
        means, logstds = self.get_means_logstds(s)
        normal = torch.distributions.Normal(means, torch.exp(logstds))
        raw_action = normal.rsample()  # uses the reparametrization trick to allow differentiability
        squashed_action = torch.tanh(raw_action)
        scale = 0.5 * (self.upper_a_bounds-self.lower_a_bounds)
        bias = scale + self.lower_a_bounds
        scaled_action = scale * squashed_action + bias

        # now need to compute log probs of scaled actions taking into account the tanh + affine transformation
        raw_logprob = normal.log_prob(raw_action).sum()
        log_det_jacobian = torch.log(scale * (1 - squashed_action ** 2) + 1e-6).sum()
        log_prob = raw_logprob - log_det_jacobian

        # get the differentiable entropy
        entropy = normal.entropy().sum() 

        return scaled_action.detach(), log_prob, entropy


class OptionManager():
    '''
    Provides all the necessary functions involving values:
        - Computes epsilon greedy policy over options
        - Computes value over options V(s) and option value upon arrival Qu(a, s, w) 
    '''
    def __init__(self, Qw_list, term_list):
        self.n_options = len(Qw_list)
        self.option_value_funcs = Qw_list
        self.termination_funcs = term_list
    
    def sample_option(self, s, epsilon):
        '''
        Uses epsilon greedy method to choose an option. Returns the index of the sampled option.
        '''
        with torch.no_grad():
            o_vals = torch.tensor([func.get_value(s).squeeze() for func in self.option_value_funcs])
        max_ind = torch.argmax(o_vals).item()
        if torch.rand(1).item() < 1 - epsilon:
            # be greedy
            return max_ind
        else:
            # uniformly explore
            return np.random.choice(list(range(self.n_options)))

    def get_quantities(self, r, s, option, epsilon, gamma, is_terminal):
        '''
        return V(s), Qu, Qw of current option, max Qw, and (differentiable) term prob in s .
        Input 'option' is the index of the current option
        Input is_terminal indicatees whether the state reaced is terminal or not
        '''
        with torch.no_grad():
            o_vals = torch.tensor([func.get_value(s).squeeze() for func in self.option_value_funcs])
        
        # get Qw of current option
        qw = o_vals[option]

        # get termination probability in s
        term_prob = self.termination_funcs[option].get_term_prob(s).squeeze()
        detached_prob = term_prob.detach()

        # compute Qu
        max_val = o_vals.max()
        Qu = r + gamma*((1-detached_prob)*qw + detached_prob*max_val)*(1-is_terminal)

        # compute V
        max_ind = torch.argmax(o_vals).item()
        probs = (epsilon/self.n_options) * torch.ones_like(o_vals)
        probs[max_ind] += 1 - epsilon
        V = (probs * o_vals).sum()

        return V, Qu, qw, max_val, term_prob
