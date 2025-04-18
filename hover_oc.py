import gymnasium as gym
import time
import PyFlyt.gym_envs  
import numpy as np
import torch
from algorithms import OptionCritic
import copy

env = gym.make('PyFlyt/QuadX-Hover-v4', sparse_reward=False)
n_options = 1
batch_size = 64
update_freq = 1000
n_epochs = 10
n_steps = 500000

OC = OptionCritic(n_options, env, epsilon=0.15, gamma=0.99, h_dim=128, 
                  qlr=0.00001, tlr=0.00001, plr=0.00001, 
                  use_buffer=True, batch_size=batch_size, horizon=update_freq)
rewards = []
end_steps = []
obs, info = env.reset()
obs = torch.from_numpy(obs).to(torch.float32)
w_index = OC.sample_option(obs)

tot_reward = 0
penalty = 0 # penalty incurred for terminating options
t0 = time.time()
for step in range(n_steps):
    # get policy outputs
    action = OC.get_action(obs, w_index)
    action = action.squeeze(0)

    # take a step
    next_obs, r, term, trunc, _ = env.step(np.array(action))
    r -= penalty
    #OC.big_buffer.add_entry(obs, next_obs, action, r, term)
    OC.buffers[w_index].add_entry(obs, next_obs, action, r, term)
    tot_reward += r
    done = term or trunc

    if (step+1) % update_freq == 0:
        for i in range(0, n_options):
            #OC.old_pols[i].mlp.load_state_dict(OC.pols[i].mlp.state_dict())
            #OC.old_pols[i].policy.load_state_dict(OC.pols[i].policy.state_dict())
            OC.old_pols[i].load_state_dict(OC.pols[i].state_dict())
        for w in range(n_options):
            for k in range(n_epochs):
                # OC.batch_update(w)
                # OC.bufferUpdateQ(w)
                #OC.epoch_update(w)
                OC.ppo_update(w)
            OC.buffers[w].empty()

    if not done:
        next_obs = torch.from_numpy(next_obs).to(torch.float32)
        with torch.no_grad():
            termprob = OC.tfuncs[w_index].get_term_prob(next_obs).squeeze()
        
        # decide if its time to re sample an option
        penalty = 0
        if np.random.rand() < termprob:
            w_index = OC.sample_option(next_obs)
            penalty = OC.xi
        
        obs = next_obs
    else:
        rewards.append(tot_reward)
        end_steps.append(step)
        tot_reward = 0
        obs, info = env.reset()
        obs = torch.from_numpy(obs).to(torch.float32)
        penalty = 0
        w_index = OC.sample_option(obs)
    
    if step % (n_steps // 1000) == 0 or step == n_steps - 1:
        # save results
        elapsed_time = time.time() - t0
        print(f'saving at step {step} after {elapsed_time / 60} minutes')
        np.savez('../data/hover_oc/test_run.npz', rewards=np.array(rewards), end_steps=np.array(end_steps))
        #np.save('../data/hover_oc/Policy_loss_overtime', OC.pol_loss_over_time)
        #np.save('../data/hover_oc/Q_loss_overtime', OC.Q_loss_over_time)
        #np.save('../data/hover_oc/Termination_loss_overtime', OC.term_loss_over_time)