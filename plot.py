import numpy as np
import matplotlib.pyplot as plt

data = np.load('evaluations.npz')
steps = data['timesteps']
rewards = data['results']
plt.plot(steps, rewards.mean(axis=1))
plt.savefig('../plots/hover_reward.png')