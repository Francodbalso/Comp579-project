import numpy as np
import matplotlib.pyplot as plt

def bootstrap_to_size(steps, rewards, stride):
    low_step, high_step = min([x.min() for x in steps]), min([x.max() for x in steps])
    bins = np.arange(low_step, high_step, stride)
    avg_rewards = np.zeros(len(bins))
    stds = np.zeros(len(bins))
    for i in range(len(bins)):
        points = []
        for j in range(len(steps)):
            xa, ya = steps[j], rewards[j]
            rets = ya[np.logical_and(low_step+i*stride <= xa, xa <= low_step+(i+1)*stride)]
            if len(rets > 0):
                points.append(rets)
        
        points = np.concatenate(points)
        avg_rewards[i] = points.mean()
        stds[i] = points.std()

    return bins, avg_rewards, stds

def pointwise_avg(steps, rewards):
    min_len = min([len(x) for x in steps])
    eps = np.arange(min_len)
    rews = np.stack([rewards[i][:min_len] for i in range(len(rewards))])
    stds = rews.std(axis=0)
    rews = rews.mean(axis=0)
    return eps, rews, stds


lr = 0.0001
savename = f'avg_waypoints_oc_plot'
plt.title('Waypoints OC (avg over 3 seeds)')
plt.ylabel('Reward')
plt.xlabel('Steps')
#plt.xlabel('Episodes')

for n_options in [1, 2, 4]:
    step_list, reward_list = [], []
    for seed in range(1, 4):
        name = 'SB3_multiQ_waypoints_all_lr'+str(lr)+'_nopts'+str(n_options)+'_seed'+str(seed)
        path= f'../data/waypoints/{name}.npz'
        data = np.load(path)
        step_list.append(data['end_steps'])
        reward_list.append(data['rewards'])
    
    # eps, avg_rewards, stds = pointwise_avg(step_list, reward_list)
    # plt.plot(eps, avg_rewards, label=f'options={n_options}')
    steps, avg_rewards, stds = bootstrap_to_size(step_list, reward_list, 3000)
    plt.plot(steps, avg_rewards, label=f'options={n_options}')

plt.legend(loc='best')
plt.savefig(f'../plots/{savename}.png')
    