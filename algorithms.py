import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from models import Qw, TerminationFunction, IntraOptionPolicy, OptionManager
from replay_buffer import ReplayBuffer

class OptionCritic():
    '''
    classic option critic algorithm that uses the following additional techniques:
        - adds baseline Qw to the intra policy gradient update
        - uses an entropy term to regularize intra option policies
        - adds small offset to advantage to discourage shrinking of options
        - uses replay buffer for the learned state option value function 
    '''
    def __init__(self, n_options, env, epsilon=0.05, gamma=0.99, h_dim=128, entropy_weight=0.01, xi=0.01, qlr=0.001, tlr=0.001, plr=0.001, use_buffer=False, batch_size=64):
        action_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        print("Observation dimension:", obs_dim)
        if use_buffer:
            self.buffer = ReplayBuffer(50000, obs_dim)
            self.MSE = torch.nn.MSELoss(reduction = 'mean')
        else:
            self.buffer = None
            self.MSE = None
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.xi = xi # for option shrinkage regularization
        self.entropy_weight = entropy_weight

        self.qfuncs = [Qw(obs_dim, h_dim) for i in range(n_options)]
        self.qfunc_optims = [torch.optim.SGD(m.mlp.parameters(), lr=qlr, weight_decay=0.01) for m in self.qfuncs]
        self.tfuncs = [TerminationFunction(obs_dim, h_dim) for i in range(n_options)]
        self.tfunc_optims = [torch.optim.SGD(m.mlp.parameters(), lr=tlr, weight_decay=0.01) for m in self.tfuncs]
        self.pols = [IntraOptionPolicy(obs_dim, h_dim, action_dim, env.action_space) for i in range(n_options)]
        self.pol_optims = [torch.optim.SGD(m.mlp.parameters(), lr=plr, weight_decay=0.01) for m in self.pols]

        self.option_manager = OptionManager(self.qfuncs, self.tfuncs)
    
    def sample_option(self, s):
        '''assume already normalized state'''
        return self.option_manager.sample_option(s, self.epsilon)
    
    def get_action_logprob_entropy(self, s, w_index):
        '''assume normalized state'''
        action, logprob, entropy = self.pols[w_index].get_action_logprob_entropy(s)
        return action, logprob, entropy

    def bufferUpdateQ(self, w_index):
        s, next_s, r, done = self.buffer.sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.float32, requires_grad=True)
        next_s = torch.tensor(next_s, dtype=torch.float32, requires_grad=True)
        r = torch.tensor(r, dtype=torch.float32).detach()
        done = torch.tensor(done, dtype=torch.float32).detach().squeeze(-1)
        
        current_qw = self.qfuncs[w_index].get_value(s)
        #print(next_s)
        with torch.no_grad():
            o_vals = [func.get_value(next_s).squeeze() for func in self.option_manager.option_value_funcs]
            o_vals = torch.stack(o_vals).squeeze(-1)
            #print(o_vals.shape)
        #print(o_vals)
        max_qs = torch.max(o_vals, dim=0)
        # print(max_qs.values)
        # print(1-done)
        # print(torch.dot(max_qs.values, (1-done)))
        # print(r)
        target = r + self.gamma * torch.dot(max_qs.values, (1-done))
        
        #print("target", target)
        #print("current qw", current_qw)
        q_loss = self.MSE(current_qw, target)
        q_loss.backward()
        self.qfunc_optims[w_index].step()
        self.qfunc_optims[w_index].zero_grad()

    def update(self, r, s, next_s, logprob, entropy, w_index, is_terminal): 
        '''
        Assume normalized next state.
        is_terminal is a boolean indiacting whether we reached a terminal state
        Returns the termination probability for next_s
        ''' 
        if self.buffer != None:
            self.buffer.add_entry(s, next_s, r, is_terminal)
        # compute values
        V, Qu, next_qw, max_q, termprob = self.option_manager.get_quantities(r, next_s, w_index, self.epsilon, self.gamma, is_terminal)
        current_qw = self.qfuncs[w_index].get_value(s).squeeze()

        # update intraoption policy
        pol_loss = (logprob*(Qu - current_qw.detach()) + self.entropy_weight*entropy)
        pol_loss.backward()
        self.pol_optims[w_index].step()
        self.pol_optims[w_index].zero_grad()

        # update termination function
        term_loss = termprob * (next_qw - V + self.xi)
        term_loss.backward()
        self.tfunc_optims[w_index].step()
        self.tfunc_optims[w_index].zero_grad()

        # update Q function
        if self.buffer != None and (self.buffer.cur_ind > self.batch_size or self.buffer.is_full):
            self.bufferUpdateQ(w_index)
        else:
            target = r + self.gamma * max_q*(1-is_terminal)
            q_loss = (target - current_qw)**2
            q_loss.backward()
            self.qfunc_optims[w_index].step()
            self.qfunc_optims[w_index].zero_grad()
        return termprob.item()
