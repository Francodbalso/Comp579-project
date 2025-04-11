import numpy as np
import matplotlib.pyplot as plt

data = np.load('../data/hover_oc/test_run.npz')
steps = data['end_steps']
rewards = data['rewards']
plt.plot(steps, rewards)
plt.xlabel('Step')
plt.ylabel('Reward')
plt.title('Hover OC')
plt.savefig('../plots/hover_oc_reward_test.png')