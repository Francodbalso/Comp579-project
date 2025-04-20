import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from models import Qw, TerminationFunction, IntraOptionPolicy, OptionManager
from replay_buffer import ReplayBuffer
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.utils import constant_fn

import torch.nn.functional as F

class OptionCritic():
    '''
    classic option critic algorithm that uses the following additional techniques:
        - adds baseline Qw to the intra policy gradient update
        - uses an entropy term to regularize intra option policies
        - adds small offset to advantage to discourage shrinking of options
        - uses replay buffer for the learned state option value function 
    '''
    def __init__(self, n_options, env, epsilon=0.1, gamma=0.99, h_dim=128, entropy_weight=0.01, xi=0.01, 
                 qlr=0.0001, tlr=0.0001, plr=0.0001, batch_size=64, horizon=1000):
        
        action_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]

        #self.big_buffer = ReplayBuffer(10*horizon, obs_dim, action_dim)
        self.buffers = [ReplayBuffer(horizon, obs_dim, action_dim) for i in range(n_options)]
        self.MSE = torch.nn.MSELoss(reduction = 'mean')
            
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.xi = xi # for option shrinkage regularization
        self.entropy_weight = entropy_weight

        self.qfuncs = [Qw(obs_dim, h_dim) for i in range(n_options)]
        self.qfunc_optims = [torch.optim.Adam(m.mlp.parameters(), lr=qlr, weight_decay=0.001) for m in self.qfuncs]
        self.tfuncs = [TerminationFunction(obs_dim, h_dim) for i in range(n_options)]
        self.tfunc_optims = [torch.optim.Adam(m.mlp.parameters(), lr=tlr, weight_decay=0.001) for m in self.tfuncs]
        #self.pols = [IntraOptionPolicy(obs_dim, h_dim, action_dim, env.action_space) for i in range(n_options)]
        #self.pol_optims = [torch.optim.Adam(m.mlp.parameters(), lr=plr, weight_decay=0.001) for m in self.pols]
        #UNCOMMENT LINEs BELOW FOR SB3
        policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[]))  # two 128‐unit hidden layers for the actor, no vf head
        self.pols = [MlpPolicy(observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: plr,
            net_arch=policy_kwargs['net_arch']) for _ in range(n_options)]
        for m in self.pols:
            if isinstance(m, nn.Linear):
                gain = 0.01 if m.out_features == self.action_dim * 2 else nn.init.calculate_gain("relu")
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0.0)
        #UNCOMMENT LINEs ABOVE FOR SB3
        self.option_manager = OptionManager(self.qfuncs, self.tfuncs)
        # self.old_pols = [IntraOptionPolicy(obs_dim, h_dim, action_dim, env.action_space) for i in range(n_options)]
        # self.old_pols = [MlpPolicy(observation_space=env.observation_space,
        #     action_space=env.action_space,
        #     lr_schedule=lambda _: 0.00001,
        #     net_arch=policy_kwargs['net_arch']) for _ in range(n_options)]        

    def sample_option(self, s):
        '''assume already normalized state'''
        return self.option_manager.sample_option(s, self.epsilon)
    
    def get_action(self, s, w_index):
        '''assume normalized state'''
        #action = self.pols[w_index].get_action(s)
        #UNCOMMENT LINEs BELOW FOR SB3
        if s.ndim == 1:
             s = s[None, :]
        action, value, log_prob = self.pols[w_index].forward(s)
        return action.detach()
    
    def get_logprob_entropy(self, a, s, w_index):
        logprob, entropy = self.pols[w_index].get_logprob_entropy(a, s)
        return logprob, entropy
            
    def get_quantities_batch(self, rewards, states, option_index, epsilon, gamma, is_terminal):

        with torch.no_grad():
            #print(states.shape)
            #print("States:", states)
            o_vals = [func.get_value(states).squeeze() for func in self.qfuncs]
            o_vals = torch.stack(o_vals)
            
        #print("ovals:", o_vals)
        #get current qw
        qw = o_vals[option_index]
        term_prob = self.tfuncs[option_index].get_term_prob(states).squeeze()
        detached_prob = term_prob.detach()
        maxs = torch.max(o_vals, dim=0)
        
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
    

    def ppo_update(self, w_index):
        '''
        does a single epoch of updates to models using recent experience from buffer.
        returns the average losses.
        '''
        n_batches = max(1, self.buffers[w_index].cur_ind // self.batch_size) # to ensure at least one pass
        eps_clip = 0.2
        avg_pol_loss = 0
        avg_q_loss = 0
        avg_term_loss = 0
        for i in range(n_batches):
            # sample batch
            states, next_states, actions, old_qws, old_next_qws, old_max_next_qws, old_logprobs, rewards, dones = self.buffers[w_index].sample(self.batch_size)
            states = torch.tensor(states, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            actions  = torch.tensor(actions, dtype=torch.float32)
            old_qws = torch.tensor(old_qws, dtype=torch.float32)
            old_next_qws = torch.tensor(old_next_qws, dtype=torch.float32)
            old_max_next_qws = torch.tensor(old_max_next_qws, dtype=torch.float32)
            old_logprobs = torch.tensor(old_logprobs, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            dones = torch.tensor(dones, dtype = torch.float32)
            
            # compute necessary quantities
            current_qws = self.qfuncs[w_index].get_value(states).squeeze()
            #logprobs, entropies = self.pols[w_index].get_logprob_entropy(actions, states)
            _, logprobs, entropies = self.pols[w_index].evaluate_actions(states, actions)
            Vs, Qus, next_qws, max_qs, termprobs = self.option_manager.get_quantities_batch(rewards, next_states, w_index, self.epsilon, self.gamma, dones)

            # update policy
            ratios = (logprobs-old_logprobs).exp()
            advantages = (Qus - current_qws.detach())
            # normalize advantages for more stable updates
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            ppo_loss = -torch.min(surrogate1, surrogate2)
            entropy_bonus = self.entropy_weight*entropies

            pol_loss = ppo_loss - entropy_bonus
            pol_loss = pol_loss.mean()
            avg_pol_loss += pol_loss.item()
            pol_loss.backward()
            #nn.utils.clip_grad_norm_(self.pols[w_index].mlp.parameters(), 0.5)
            #self.pol_optims[w_index].step()
            #self.pol_optims[w_index].zero_grad()
            #UNCOMMENT LINEs BELOW FOR SB3
            nn.utils.clip_grad_norm_(self.pols[w_index].parameters(), 0.5)
            self.pols[w_index].optimizer.step()
            self.pols[w_index].optimizer.zero_grad()
            

            # update termination
            advantages = next_qws - Vs + self.xi
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # normalization here too
            term_loss = termprobs * advantages
            term_loss = term_loss.mean()
            avg_term_loss += term_loss.item()
            term_loss.backward()
            nn.utils.clip_grad_norm_(self.tfuncs[w_index].mlp.parameters(), 0.5)
            self.tfunc_optims[w_index].step()
            self.tfunc_optims[w_index].zero_grad()

            # update Q function using clipped loss 
            target = rewards + self.gamma * ((1-termprobs.detach()) * old_next_qws + termprobs.detach() * old_max_next_qws) * (1-dones)
            q_clipped = old_qws + torch.clamp(current_qws - old_qws, -eps_clip, eps_clip)
            loss_unclipped = (current_qws - target)**2
            loss_clipped = (q_clipped - target)**2
            q_loss = 0.5 * torch.max(loss_unclipped, loss_clipped).mean()

            avg_q_loss += q_loss.item()
            q_loss.backward()
            nn.utils.clip_grad_norm_(self.qfuncs[w_index].mlp.parameters(), 0.5)
            self.qfunc_optims[w_index].step()
            self.qfunc_optims[w_index].zero_grad()
        
        return avg_pol_loss/n_batches, avg_term_loss/n_batches, avg_q_loss/n_batches
