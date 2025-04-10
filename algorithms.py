import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from models import Qw, TerminationFunction, IntraOptionPolicy, OptionManager

class OptionCritic():
    '''
    classic option critic algorithm that uses the following additional techniques:
        - adds baseline Qw to the intra policy gradient update
        - uses an entropy term to regularize intra option policies
        - adds small offset to advantage to discourage shrinking of options
        - uses replay buffer for the learned state option value function 
    '''
    def __init__(self, n_options, env, epsilon=0.1, gamma=0.99, h_dim=128, qlr=0.001, tlr=0.001, plr=0.001):
        action_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]

        self.epsilon = epsilon
        self.gamma = gamma

        self.qfuncs = [Qw(obs_dim, h_dim) for i in range(n_options)]
        self.qfunc_optims = [torch.optim.SGD(m.mlp.parameters(), lr=qlr) for m in self.qfuncs]
        self.tfuncs = [TerminationFunction(obs_dim, h_dim) for i in range(n_options)]
        self.tfunc_optims = [torch.optim.SGD(m.mlp.parameters(), lr=tlr) for m in self.tfuncs]
        self.pols = [IntraOptionPolicy(obs_dim, h_dim, action_dim, env.action_space) for i in range(n_options)]
        self.pol_optims = [torch.optim.SGD(m.mlp.parameters(), lr=plr) for m in self.pols]

        self.option_manager = OptionManager(self.qfuncs, self.tfuncs)
    
    def sample_option(self, s):
        '''assume already normalized state'''
        return self.option_manager.sample_option(s, self.epsilon)
    
    def get_action_logprob_entropy(self, s, w_index):
        '''assume normalized state'''
        action, logprob, entropy = self.pols[w_index].get_action_logprob_entropy(s)
        return action, logprob, entropy

    def update(self, r, next_s, logprob, entropy, w_index): 
        '''assume normalized next state''' 
        # compute value over options and value upon arrival
        V, Qu = self.option_manager.get_V_Qu(r, next_s, w_index, self.epsilon, self.gamma)

        # update intraoption policy
        pol_loss = -logprob*Qu + entropy
        pol_loss.backward()
        self.pol_optims[w_index].step()
        self.pol_optims[w_index].zero_grad()

        # update termination function
