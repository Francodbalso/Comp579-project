import gymnasium as gym
import time
import PyFlyt.gym_envs  
import numpy as np
import torch
from algorithms import OptionCritic
import copy
from PyFlyt.gym_envs import FlattenWaypointEnv

seed_num = 1
torch.manual_seed(seed_num)
#savepath = 'hover_oc/test_run'
env = gym.make('PyFlyt/QuadX-Hover-v4', sparse_reward=False)
# env = gym.make('PyFlyt/QuadX-Waypoints-v4')
# env = FlattenWaypointEnv(env, context_length=2)
print("Observation space", env.observation_space.shape[0])

for qlr in [0.0001, 0.00001, 0.000001]:
    for n_options in [1, 2, 4]:
        savepath = 'hover_oc/SB3_1Q_hover_'+str(qlr)+'_'+str(n_options)+'_seed'+str(seed_num)
        # env = gym.make('InvertedPendulum-v5')
        #n_options = n_options
        batch_size = 128
        update_freq = 2000
        n_epochs = 10
        n_steps = 500000
        one_hots = [torch.nn.functional.one_hot(torch.tensor(i), num_classes=n_options).to(torch.float32) for i in range(n_options)]

        OC = OptionCritic(n_options, env, epsilon=0.15, gamma=0.99, h_dim=128, 
                        qlr=qlr, tlr=0.00001, plr=0.00001, 
                        batch_size=batch_size, horizon=update_freq+1)
        rewards = []
        end_steps = []
        update_steps = []
        avg_pol_loss = []
        avg_term_loss = []
        avg_q_loss = []
        obs, info = env.reset()
        obs = torch.from_numpy(obs).to(torch.float32)
        w_index = OC.sample_option(obs)

        tot_reward = 0
        penalty = 0 # penalty incurred for terminating options
        t0 = time.time()
        for step in range(n_steps):
            if step%1000 == 0:
                print(step)
            # get policy outputs
            action = OC.get_action(obs, w_index)

            # take a step
            next_obs, r, term, trunc, _ = env.step(np.array(action.squeeze(0)))
            next_obs = torch.from_numpy(next_obs).to(torch.float32)
            r -= penalty

            # log things
            with torch.no_grad():
                # doing all these now to save computation inside the update loops
                #logprob, _ = OC.pols[w_index].get_logprob_entropy(action, obs[None, :])
                #UNCOMMENT LINE BELOW FOR SB3
                _, logprob, _ = OC.pols[w_index].evaluate_actions(obs[None, :], action[None, :])
                #print(logprob.sum())
                logprob = logprob.sum()
                #logprob = logprob.squeeze().item()
                qw = OC.qfunc.get_value(torch.cat((obs, one_hots[w_index]))).squeeze().item()
                next_qws = torch.tensor([OC.qfunc.get_value(torch.cat((obs, o_hot))).squeeze().item() for o_hot in one_hots])
                #qw = OC.qfuncs[w_index].get_value(obs).squeeze().item()
                #next_qws = [OC.qfuncs[i].get_value(next_obs).squeeze().item() for i in range(n_options)]
                next_qw = next_qws[w_index]
                max_next_qw = max(next_qws)
            # print("qw:", qw)
            # print("next_qw:", next_qw)
            # print("max_next_qws:", max_next_qw)
            OC.buffers[w_index].add_entry(obs, next_obs, action.squeeze(0), qw, next_qw, max_next_qw, logprob, r, term)
            tot_reward += r
            done = term or trunc

            if (step+1) % update_freq == 0:
                # computing average losses during training
                pol_loss = 0
                term_loss = 0
                q_loss = 0
                for w in range(n_options):
                    for k in range(n_epochs):
                        pl, tl, ql = OC.ppo_update(w)
                        pol_loss += pl; term_loss += tl; q_loss += ql
                    OC.buffers[w].empty()

                avg_pol_loss.append(pol_loss/(n_options*n_epochs))
                avg_term_loss.append(term_loss/(n_options*n_epochs))
                avg_q_loss.append(q_loss/(n_options*n_epochs))
                update_steps.append(step)

            if not done:
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
            
            if step % (n_steps // 10) == 0 or step == n_steps - 1:
                # save results
                elapsed_time = time.time() - t0
                print(f'saving at step {step} after {elapsed_time / 60} minutes')
                np.savez(f'../data/{savepath}.npz', 
                        rewards=np.array(rewards), 
                        end_steps=np.array(end_steps),
                        update_steps=np.array(update_steps),
                        pol_loss=np.array(avg_pol_loss),
                        q_loss=np.array(avg_q_loss),
                        term_loss=np.array(avg_term_loss)
                        )
        
