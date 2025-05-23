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
    '''
    def __init__(self, s_dim, h_dim):
        self.mlp = MLP(s_dim, h_dim, 1)
    
    def get_value(self, s):
        # assume already normalized state
        return self.mlp(s)

class SingleQw():
    '''
    The option value model Q_U(s, w) from the paper.
    '''
    def __init__(self, s_dim, emb_dim, h_dim, n_options):
        self.n_options = n_options
        if n_options == 1:
            self.mlp = MLP(s_dim, h_dim, 1)
        else:
            self.embedding = nn.Embedding(n_options, emb_dim)
            self.mlp = MLP(s_dim + emb_dim, h_dim, 1)
    
    def get_value(self, s, w):
        if self.n_options == 1:
            return self.mlp(s)
        else:
            w = torch.tensor(w)
            if s.dim() != 1:
                w = w.unsqueeze(0).expand(len(s))

            emb = self.embedding(w)
            inp = torch.cat([s, emb], dim=-1)
            return self.mlp(inp)
    
    def parameters(self):
        if self.n_options == 1:
            return self.mlp.parameters()
        else:
            return list(self.embedding.parameters()) + list(self.mlp.parameters())
    

class TerminationFunction():
    '''
    The termination model which takes in a state and returns a probability of termination.
    Again each instance of this class will correspond to a specific option.
    in_dim should be the state dimension.
    '''
    def __init__(self, state_dim, h_dim):
        self.mlp = MLP(state_dim, h_dim, 1)
        desired_prob = 0.2  # or whatever you'd like
        initial_bias = torch.logit(torch.tensor(desired_prob))
        # Access the final linear layer and set its bias
        self.mlp.layers[-1].bias.data.fill_(initial_bias.item())
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

        # this is for the tanh action squashing
        self.scale = (0.5 * (self.upper_a_bounds-self.lower_a_bounds)).unsqueeze(0)
        self.bias = self.scale + self.lower_a_bounds.unsqueeze(0)
        
        # make an mlp with last layer's weights extra small initially to help tanh gradients
        self.mlp = MLP(state_dim, h_dim, 2 * action_dim)
        self.mlp.layers[-1].weight.data *= 0.1
        self.mlp.layers[-1].bias.data *= 0.1
    
    def get_means_logstds(self, s):
        '''expecting s to be of shape (batch, state_dim)'''
        # first half of outputs will be the means, other half will be logstds
        if s.dim() == 1:
            s = s.unsqueeze(0)
        outputs = self.mlp(s)
        #print(outputs)
        means = outputs[:, :self.action_dim]
        # bound the log stds, using tanh, to avoid numerical instabilities later on
        # can also change the slope of the tanh to avoid needing to make weights small initially
        lowerbound, upperbound = -3, 2
        logstds = outputs[:, self.action_dim:]
        logstds = lowerbound + 0.5 * (upperbound - lowerbound) * (torch.tanh(logstds) + 1) 
        return means, logstds

    def get_action(self, s):
        '''
        expecting s to be of shape (batch, state_dim)
        returns actions in shape (batch, action_dim)
        '''
        if s.dim() == 1:
            s = s.unsqueeze(0)
        with torch.no_grad():
            means, logstds = self.get_means_logstds(s)
        
        print('means: ', means, ' stds: ', torch.exp(logstds))
        normal = torch.distributions.Normal(means, torch.exp(logstds))
        raw_action = normal.sample()  
        squashed_action = torch.tanh(raw_action)
        scaled_action = self.scale * squashed_action + self.bias

        return scaled_action

    def get_logprob_entropy(self, a, s):
        '''
        expecting a and s to have a batch dimension
        returns vector of log_probs and vector of entropies each of dimension (batch,)
        '''
        if s.dim() == 1:
            s = s.unsqueeze(0)
        means, logstds = self.get_means_logstds(s)
        #print(means, logstds)
        normal = torch.distributions.Normal(means, torch.exp(logstds))

        # now need to compute log probs of scaled actions taking into account the tanh + affine transformation
        squashed_action = (a - self.bias) / self.scale
        raw_action = torch.atanh(torch.clamp(squashed_action, -0.999, 0.999))
        raw_logprob = normal.log_prob(raw_action).sum(dim=1)
        log_det_jacobian = torch.log(self.scale * (1 - squashed_action ** 2 + 1e-6) + 1e-6).sum(dim=1)
        log_prob = raw_logprob - log_det_jacobian

        # get the differentiable entropy
        entropy = normal.entropy().sum(dim=1) 

        return log_prob, entropy


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
    def get_Values_batch(self, states, epsilon):
        with torch.no_grad():
            #print(states.shape)
            #print("States:", states)
            o_vals = [func.get_value(states).squeeze() for func in self.option_value_funcs]
            o_vals = torch.stack(o_vals)
        maxs = torch.max(o_vals, dim=0)

        probs = (epsilon/self.n_options) * torch.ones_like(o_vals).squeeze(-1)
        for batch_idx in range(o_vals.shape[1]):
            probs[maxs.indices[batch_idx], batch_idx] = 1 - epsilon + (epsilon/self.n_options)
        V = probs*o_vals.squeeze(-1)
        #print(V)
        V = V.sum(dim=0)
        return V
    
    def get_quantities_batch(self, rewards, states, option_index, epsilon, gamma, is_terminal):

        with torch.no_grad():
            #print(states.shape)
            #print("States:", states)
            o_vals = [func.get_value(states).squeeze() for func in self.option_value_funcs]
            o_vals = torch.stack(o_vals)
            
        #print("ovals:", o_vals)
        #get current qw
        qw = o_vals[option_index]
        term_prob = self.termination_funcs[option_index].get_term_prob(states).squeeze()
        detached_prob = term_prob.detach()
        # print("detached probs:", detached_prob)
        # print("qw:", qw)
        #o_vals[0][0] = 100
        #o_vals[0][1] = 99
        #o_vals[1][0] = 199
        maxs = torch.max(o_vals, dim=0)
        #print("maxs:", maxs)
        #print("(1-detached_prob)*qw ", (1-detached_prob)*qw )
        #print("detached_prob*maxs.values", detached_prob*maxs.values)
        #print("(1-is_terminal)", (1-is_terminal))
        Qu = rewards.squeeze() + gamma*((1-detached_prob)*qw + detached_prob*maxs.values)*(1-is_terminal).squeeze()
        # print("QU:", Qu)

        probs = (epsilon/self.n_options) * torch.ones_like(o_vals).squeeze(-1)
        for batch_idx in range(o_vals.shape[1]):
            probs[maxs.indices[batch_idx], batch_idx] = 1 - epsilon + (epsilon/self.n_options)

        V = probs*o_vals.squeeze(-1)
        #print(V)
        V = V.sum(dim=0)
        #print(V)
        return V, Qu, qw, maxs.values, term_prob
    



class SingleQOptionManager():
    '''
    Provides all the necessary functions involving values:
        - Computes epsilon greedy policy over options
        - Computes value over options V(s) and option value upon arrival Qu(a, s, w) 
    '''
    def __init__(self, Qw, term_list):
        self.n_options = len(term_list)
        self.qw = Qw
        self.termination_funcs = term_list
        #self.one_hots = [torch.nn.functional.one_hot(torch.tensor(i), num_classes=self.n_options).to(torch.float32) for i in range(self.n_options)]
        
    def sample_option(self, s, epsilon):
        '''
        Uses epsilon greedy method to choose an option. Returns the index of the sampled option.
        '''
        with torch.no_grad():
            o_vals = torch.tensor([self.qw.get_value(s, w).squeeze() for w in range(self.n_options)])
        max_ind = torch.argmax(o_vals).item()
        if torch.rand(1).item() < 1 - epsilon:
            # be greedy
            return max_ind
        else:
            # uniformly explore
            return np.random.choice(list(range(self.n_options)))
    
    
    def get_quantities_batch(self, rewards, states, option_index, epsilon, gamma, is_terminal):

        with torch.no_grad():
            # o_vals = []
            # #print(states.shape)
            # #print("States:", states)
            # for one_hot_vector in self.one_hots:
            #     one_hot_expanded = one_hot_vector.unsqueeze(0).expand(states.shape[0], -1)  # shape [64, 2]
            #     augmented_batch = torch.cat((states, one_hot_expanded), dim=1)  # shape [64, 5]
            #     o_vals.append(self.qw.get_value(augmented_batch).squeeze())
            o_vals = [self.qw.get_value(states, w).squeeze(0) for w in range(self.n_options)]
            o_vals = torch.stack(o_vals)
            #print("out values shape:", o_vals.shape)
            
        #print("ovals:", o_vals)
        #get current qw
        qw = o_vals[option_index]
        term_prob = self.termination_funcs[option_index].get_term_prob(states).squeeze()
        detached_prob = term_prob.detach()
        # print("detached probs:", detached_prob)
        # print("qw:", qw)
        #o_vals[0][0] = 100
        #o_vals[0][1] = 99
        #o_vals[1][0] = 199
        maxs = torch.max(o_vals, dim=0)
        #print("maxs:", maxs)
        #print("(1-detached_prob)*qw ", (1-detached_prob)*qw )
        #print("detached_prob*maxs.values", detached_prob*maxs.values)
        #print("(1-is_terminal)", (1-is_terminal))
        Qu = rewards.squeeze() + gamma*((1-detached_prob)*qw + detached_prob*maxs.values)*(1-is_terminal).squeeze()
        # print("QU:", Qu)

        probs = (epsilon/self.n_options) * torch.ones_like(o_vals)
        for batch_idx in range(o_vals.shape[1]):
            probs[maxs.indices[batch_idx], batch_idx] = 1 - epsilon + (epsilon/self.n_options)

        V = probs*o_vals
        #print(V)
        V = V.sum(dim=0)
        #print(V)
        return V, Qu, qw, maxs.values, term_prob