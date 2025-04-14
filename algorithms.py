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
    def __init__(self, n_options, env, epsilon=0.05, gamma=0.99, h_dim=128, entropy_weight=0.01, xi=0.01, qlr=0.001, tlr=0.001, plr=0.001, use_buffer=False, batch_size=64, horizon=1000):
        action_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]

        if use_buffer:
            self.big_buffer = ReplayBuffer(5*horizon, obs_dim, action_dim)
            self.buffers = [ReplayBuffer(horizon, obs_dim, action_dim) for i in range(n_options)]
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
        self.qfunc_optims = [torch.optim.SGD(m.mlp.parameters(), lr=qlr, weight_decay=0.001) for m in self.qfuncs]
        self.tfuncs = [TerminationFunction(obs_dim, h_dim) for i in range(n_options)]
        self.tfunc_optims = [torch.optim.SGD(m.mlp.parameters(), lr=tlr, weight_decay=0.001) for m in self.tfuncs]
        self.pols = [IntraOptionPolicy(obs_dim, h_dim, action_dim, env.action_space) for i in range(n_options)]
        self.pol_optims = [torch.optim.SGD(m.mlp.parameters(), lr=plr, weight_decay=0.001) for m in self.pols]
        self.option_manager = OptionManager(self.qfuncs, self.tfuncs)
    
    def sample_option(self, s):
        '''assume already normalized state'''
        return self.option_manager.sample_option(s, self.epsilon)
    
    def get_action_logprob_entropy(self, s, w_index):
        '''assume normalized state'''
        action, logprob, entropy = self.pols[w_index].get_action_logprob_entropy(s)
        return action, logprob, entropy

    def bufferUpdateQ(self, w_index):
        if self.buffers[w_index] == None or not (self.buffers[w_index].cur_ind > self.batch_size or self.buffers[w_index].is_full):
            return
        s, next_s, actions, r, done = self.buffers[w_index].sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.float32, requires_grad=True)
        next_s = torch.tensor(next_s, dtype=torch.float32, requires_grad=True)
        r = torch.tensor(r, dtype=torch.float32).detach().squeeze()
        done = torch.tensor(done, dtype=torch.float32).detach().squeeze()
        
        current_qw = self.qfuncs[w_index].get_value(s).squeeze()
        #print(next_s)
        with torch.no_grad():
            o_vals = [func.get_value(next_s).squeeze() for func in self.qfuncs]
            o_vals = torch.stack(o_vals)
            #print(o_vals.shape)
        # print("++++++++++++++++++++++++++++++PRINTS IN BUFFER UPDAT Q++++++++++++++++++++++++++++++")
        # print(o_vals)
        max_qs = torch.max(o_vals, dim=0)
        # print(max_qs.values)
        # print(1-done)
        # print(torch.dot(max_qs.values, (1-done)))
        # print(r)
        target = r + self.gamma * max_qs.values*(1-done)
        
        # print("target", target)
        # print("current qw", current_qw)
        # print("++++++++++++++++++++++++++++++END++++++++++++++++++++++++++++++")
        q_loss = self.MSE(current_qw, target)
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.qfuncs[w_index].mlp.parameters(), 1.0)
        self.qfunc_optims[w_index].step()
        self.qfunc_optims[w_index].zero_grad()

    def batch_update(self, w_index):
        if self.big_buffer == None or not(self.big_buffer.cur_ind > self.batch_size or self.big_buffer.is_full):
            return 
        states, next_states, actions, rewards, dones = self.big_buffer.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32, requires_grad=True)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions  = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).detach()
        dones = torch.tensor(dones, dtype = torch.float32).detach()
        # print("states:", states)
        # print("next_states:", next_states)
        # print("actions:", actions)
        # print("rewards:", rewards)
        # print("dones:", dones)

        current_qws = self.qfuncs[w_index].get_value(states)
        logprobs, entropies = self.pols[w_index].get_logprob_entropy(actions, states)
        Vs, Qus, next_qws, max_qs, termprobs = self.option_manager.get_quantities_batch(rewards, next_states, w_index, self.epsilon, self.gamma, dones)
        # print("Vs:", Vs)
        # print("Qus:", Qus)
        # print("next_qws:", next_qws)
        # print("max_qs:", max_qs)
        # print("termprobs:", termprobs)

        pol_loss = (logprobs*(Qus - current_qws.detach()) + self.entropy_weight*entropies)
        #print(pol_loss)
        pol_loss = pol_loss.sum()
        print("Pooling loss:", pol_loss)
        pol_loss.backward()
        nn.utils.clip_grad_norm_(self.pols[w_index].mlp.parameters(), 1.0)
        self.pol_optims[w_index].step()
        self.pol_optims[w_index].zero_grad()
        
        term_loss = termprobs * (next_qws - Vs+ self.xi)
        term_loss = term_loss.sum()
        term_loss.backward()
        nn.utils.clip_grad_norm_(self.tfuncs[w_index].mlp.parameters(), 1.0)
        self.tfunc_optims[w_index].step()
        self.tfunc_optims[w_index].zero_grad()
            
    def update(self, r, s, next_s, logprob, entropy, w_index, is_terminal): 
        '''
        Assume normalized next state.
        is_terminal is a boolean indiacting whether we reached a terminal state
        Returns the termination probability for next_s
        ''' 
        # if self.buffer != None:
        #     self.buffer.add_entry(s, next_s, r, is_terminal)
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
        # if self.buffer != None and (self.buffer.cur_ind > self.batch_size or self.buffer.is_full):
        #     self.bufferUpdateQ(w_index)
        # else:
        target = r + self.gamma * max_q*(1-is_terminal)
        q_loss = (target - current_qw)**2
        q_loss.backward()
        self.qfunc_optims[w_index].step()
        self.qfunc_optims[w_index].zero_grad()
        return termprob.item()
