import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from models import Qw, TerminationFunction, IntraOptionPolicy, OptionManager
from replay_buffer import ReplayBuffer
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.utils import constant_fn

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
            self.big_buffer = ReplayBuffer(10*horizon, obs_dim, action_dim)
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
        #self.pols = [MlpPolicy(env.observation_space, env.action_space, lr_schedule=constant_fn(0.00001)) for i in range(n_options)]
        self.pol_optims = [torch.optim.SGD(m.mlp.parameters(), lr=plr, weight_decay=0.001) for m in self.pols]
        self.option_manager = OptionManager(self.qfuncs, self.tfuncs)
        self.old_pols = [IntraOptionPolicy(obs_dim, h_dim, action_dim, env.action_space) for i in range(n_options)]
        #self.pols = [MlpPolicy(env.observation_space, env.action_spac, lr_schedule=constant_fn(0.00001)) for i in range(n_options)]
        self.pol_loss_over_time = []
        self.Q_loss_over_time = []
        self.term_loss_over_time = []
    def sample_option(self, s):
        '''assume already normalized state'''
        return self.option_manager.sample_option(s, self.epsilon)
    
    def get_action(self, s, w_index):
        '''assume normalized state'''
        action = self.pols[w_index].get_action(s)
        return action
    
    def get_logprob_entropy(self, a, s, w_index):
        logprob, entropy = self.pols[w_index].get_logprob_entropy(a, s)
        return logprob, entropy

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
        # print("++++++++++++++++++++++++++++++PRINTS IN BUFFER UPDATE Q++++++++++++++++++++++++++++++")
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
        #print("QFunction loss:", q_loss)
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.qfuncs[w_index].mlp.parameters(), 1.0)
        self.qfunc_optims[w_index].step()
        self.qfunc_optims[w_index].zero_grad()

    def ppo_update(self, w_index):
        n_batches = self.buffers[w_index].cur_ind // self.batch_size
        eps_clip = 0.2
        for i in range(n_batches):
            # sample batch
            states, next_states, actions, rewards, dones = self.buffers[w_index].sample(self.batch_size)
            states = torch.tensor(states, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            actions  = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32).squeeze()
            dones = torch.tensor(dones, dtype = torch.float32).squeeze()
            
            # compute necessary quantities
            current_qws = self.qfuncs[w_index].get_value(states).squeeze()
            logprobs, entropies = self.pols[w_index].get_logprob_entropy(actions, states)
            Vs, Qus, next_qws, max_qs, termprobs = self.option_manager.get_quantities_batch(rewards, next_states, w_index, self.epsilon, self.gamma, dones)
            #current_Vs = self.option_manager.get_Values_batch(states, self.epsilon)
            old_logprobs, _ = self.old_pols[w_index].get_logprob_entropy(actions, states)
            # update policy
            ratios = (logprobs-old_logprobs).exp()
            advantages = (Qus - current_qws.detach())

            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            ppo_loss = -torch.min(surrogate1, surrogate2)
            entropy_bonus = self.entropy_weight*entropies

            pol_loss = ppo_loss - entropy_bonus
            pol_loss = pol_loss.mean()
            self.pol_loss_over_time.append(pol_loss.item())
            pol_loss.backward()
            #nn.utils.clip_grad_norm_(self.pols[w_index].mlp.parameters(), 1.0)
            self.pol_optims[w_index].step()
            self.pol_optims[w_index].zero_grad()

            # update termination
            term_loss = termprobs * (next_qws - Vs + self.xi)
            term_loss = term_loss.mean()
            self.term_loss_over_time.append(term_loss.item())
            term_loss.backward()
            nn.utils.clip_grad_norm_(self.tfuncs[w_index].mlp.parameters(), 1.0)
            self.tfunc_optims[w_index].step()
            self.tfunc_optims[w_index].zero_grad()

            # update Q function
            target = rewards + self.gamma * ((1-termprobs.detach()) * next_qws + termprobs.detach() * max_qs) * (1-dones)
            q_loss = self.MSE(current_qws, target)
            self.Q_loss_over_time.append(q_loss.item())
            q_loss.backward()
            nn.utils.clip_grad_norm_(self.qfuncs[w_index].mlp.parameters(), 1.0)
            self.qfunc_optims[w_index].step()
            self.qfunc_optims[w_index].zero_grad()


    def epoch_update(self, w_index):
        '''
        Does one epoch of updates for all NNs for the given option.
        This assumes that the buffers are never completely full.
        '''
        # doing approximate epochs cuz im lazy rn
        n_batches = self.buffers[w_index].cur_ind // self.batch_size
        for i in range(n_batches):
            # sample batch
            states, next_states, actions, rewards, dones = self.buffers[w_index].sample(self.batch_size)
            states = torch.tensor(states, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            actions  = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32).squeeze()
            dones = torch.tensor(dones, dtype = torch.float32).squeeze()
            
            # compute necessary quantities
            current_qws = self.qfuncs[w_index].get_value(states).squeeze()
            logprobs, entropies = self.pols[w_index].get_logprob_entropy(actions, states)
            Vs, Qus, next_qws, max_qs, termprobs = self.option_manager.get_quantities_batch(rewards, next_states, w_index, self.epsilon, self.gamma, dones)

            # update policy
            pol_loss = (logprobs*(Qus - current_qws.detach()) - self.entropy_weight*entropies)
            pol_loss = pol_loss.mean()
            pol_loss.backward()
            nn.utils.clip_grad_norm_(self.pols[w_index].mlp.parameters(), 1.0)
            self.pol_optims[w_index].step()
            self.pol_optims[w_index].zero_grad()
            
            # update termination
            term_loss = termprobs * (next_qws - Vs + self.xi)
            term_loss = term_loss.mean()
            term_loss.backward()
            nn.utils.clip_grad_norm_(self.tfuncs[w_index].mlp.parameters(), 1.0)
            self.tfunc_optims[w_index].step()
            self.tfunc_optims[w_index].zero_grad()

            # update Q function
            target = rewards + self.gamma * max_qs * (1-dones)
            q_loss = self.MSE(current_qws, target)
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

        current_qws = self.qfuncs[w_index].get_value(states).squeeze()
        logprobs, entropies = self.pols[w_index].get_logprob_entropy(actions, states)
        Vs, Qus, next_qws, max_qs, termprobs = self.option_manager.get_quantities_batch(rewards, next_states, w_index, self.epsilon, self.gamma, dones)
        # print(current_qws)
        # print("Vs:", Vs)
        # print("Qus:", Qus)
        # print("next_qws:", next_qws)
        # print("max_qs:", max_qs)
        print("termprobs:", termprobs)
        # print("log_probs", logprobs)
        # print("entropies", entropies)
        pol_loss = -(logprobs*(Qus - current_qws.detach()) + self.entropy_weight*entropies)
        #print("Policy loss", pol_loss)
        pol_loss = pol_loss.mean()
        #print("Policy loss:", pol_loss)
        pol_loss.backward()
        nn.utils.clip_grad_norm_(self.pols[w_index].mlp.parameters(), 1.0)
        self.pol_optims[w_index].step()
        self.pol_optims[w_index].zero_grad()
        
        
        term_loss = termprobs * (next_qws - Vs+ self.xi)
        
        term_loss = term_loss.mean()
        #print("Termination loss", term_loss)
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
