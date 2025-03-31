import numpy as np
import matplotlib.pyplot as plt

data = np.load('../data/waypoint_baseline/evaluations.npz')
steps = data['timesteps']
rewards = data['results']
plt.plot(steps, rewards.mean(axis=1))
plt.savefig('../plots/waypoint_baseline_reward.png')