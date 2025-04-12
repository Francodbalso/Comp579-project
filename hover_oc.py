import gymnasium as gym
import time
import PyFlyt.gym_envs
import numpy as np
import torch
from algorithms import OptionCritic

env = gym.make('PyFlyt/QuadX-Hover-v4')
n_options = 4
OC = OptionCritic(n_options, env, epsilon=0.05, gamma=0.99, h_dim=128, qlr=0.00001, tlr=0.00001, plr=0.00001)

n_steps = 200000
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
    action, logprob, entropy = OC.get_action_logprob_entropy(obs, w_index)

    # take a step
    next_obs, r, term, trunc, _ = env.step(np.array(action))
    r -= penalty
    tot_reward += r
    done = term or trunc
    
    if not done:
        next_obs = torch.from_numpy(next_obs).to(torch.float32)
        # make updates
        termprob = OC.update(r, obs, next_obs, logprob, entropy, w_index)
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
    
    if step % (n_steps // 100) == 0 or step == n_steps - 1:
        # save results
        elapsed_time = time.time() - t0
        print(f'saving at step {step} after {elapsed_time / 60} minutes')
        np.savez('../data/hover_oc/test_run.npz', rewards=np.array(rewards), end_steps=np.array(end_steps))
