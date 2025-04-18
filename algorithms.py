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
        #self.pols = [IntraOptionPolicy(obs_dim, h_dim, action_dim, env.action_space) for i in range(n_options)]
        self.pols = [MlpPolicy(env.observation_space, env.action_space, lr_schedule= lambda _: 0.0001) for i in range(n_options)]
        #self.pol_optims = [torch.optim.SGD(m.mlp.parameters(), lr=plr, weight_decay=0.001) for m in self.pols]
        self.option_manager = OptionManager(self.qfuncs, self.tfuncs)
        #self.old_pols = [IntraOptionPolicy(obs_dim, h_dim, action_dim, env.action_space) for i in range(n_options)]
        self.old_pols = [MlpPolicy(env.observation_space, env.action_space, lr_schedule= lambda _: 0.0001) for i in range(n_options)]
        self.pol_loss_over_time = []
        self.Q_loss_over_time = []
        self.term_loss_over_time = []
    def sample_option(self, s):
        '''assume already normalized state'''
        return self.option_manager.sample_option(s, self.epsilon)
    
    def get_action(self, s, w_index):
        '''assume normalized state'''
        #action = self.pols[w_index].get_action(s)
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
            o_vals = [self.pols.predict_values(states) for func in self.pols]
            o_vals = torch.stack(o_vals)
            
        #print("ovals:", o_vals)
        #get current qw
        qw = o_vals[option_index]
        term_prob = self.t_funcs[option_index].get_term_prob(states).squeeze()
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
        self.pols[w_index].train()
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
            current_s = self.pols[w_index].predict_values(states).squeeze()
            #logprobs, entropies = self.pols[w_index].get_logprob_entropy(actions, states)
            _, logprobs, entropies = self.pols[w_index].evaluate_actions(states, actions)
            Vs, Qus, next_qws, max_qs, termprobs = self.get_quantities_batch(rewards, next_states, w_index, self.epsilon, self.gamma, dones)
            #current_Vs = self.option_manager.get_Values_batch(states, self.epsilon)
            #old_logprobs, _ = self.old_pols[w_index].get_logprob_entropy(actions, states)
            old_logprobs, _, _ = self.old_pols[w_index].evaluate_actions(states, actions)

            # update policy
            ratios = (logprobs-old_logprobs).exp()
            advantages = (Qus - current_s.detach())
            target = rewards + self.gamma * ((1-termprobs.detach()) * next_qws + termprobs.detach() * max_qs) * (1-dones)
            q_loss = self.MSE(current_s, target)
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            ppo_loss = -torch.min(surrogate1, surrogate2)
            entropy_bonus = self.entropy_weight*entropies

            pol_loss = ppo_loss - entropy_bonus + q_loss
            pol_loss = pol_loss.mean()
            #self.pol_loss_over_time.append(pol_loss.item())
            self.pols[w_index].optimizer.zero_grad()
            pol_loss.backward()
            #nn.utils.clip_grad_norm_(self.pols[w_index].mlp.parameters(), 1.0)
            self.pols[w_index].optimizer.step()
            #self.pol_optims[w_index].zero_grad()

            # update termination
            term_loss = termprobs * (next_qws - Vs + self.xi)
            term_loss = term_loss.mean()
            #self.term_loss_over_time.append(term_loss.item())
            self.tfunc_optims[w_index].zero_grad()
            term_loss.backward()
            nn.utils.clip_grad_norm_(self.tfuncs[w_index].mlp.parameters(), 1.0)
            self.tfunc_optims[w_index].step()
           

            # # update Q function
            # target = rewards + self.gamma * ((1-termprobs.detach()) * next_qws + termprobs.detach() * max_qs) * (1-dones)
            # q_loss = self.MSE(current_s, target)
            # #self.Q_loss_over_time.append(q_loss.item())
            # self.qfunc_optims[w_index].zero_grad()
            # q_loss.backward()
            # nn.utils.clip_grad_norm_(self.qfuncs[w_index].mlp.parameters(), 1.0)
            # self.qfunc_optims[w_index].step()
            


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

