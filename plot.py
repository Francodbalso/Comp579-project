import numpy as np
import matplotlib.pyplot as plt



fig, ax = plt.subplots(2, 2)

data = np.load('../data/hover_oc/test_run.npz')
steps = data['end_steps']
rewards = data['rewards']
ax[0][0].plot(steps, rewards)
ax[0][0].set_xlabel('Step')
ax[0][0].set_ylabel('Reward')
ax[0][0].set_title('Hover OC')
#ax[0][0].savefig('../plots/hover_oc_reward_test.png')

data = np.load('../data/hover_oc/Policy_loss_overtime.npy')
print(data)
#steps = data['end_steps']
#rewards = data['rewards']
ax[0][1].plot(data)
ax[0][1].set_xlabel('Step')
ax[0][1].set_ylabel('Loss')
ax[0][1].set_title('Hover OC Policy Loss')

data = np.load('../data/hover_oc/Q_loss_overtime.npy')
# steps = data['end_steps']
# rewards = data['rewards']
ax[1][1].plot(data)
ax[1][1].set_xlabel('step')
ax[1][1].set_ylabel('Loss')
ax[1][1].set_title('Hover OC Q loss')

data = np.load('../data/hover_oc/Termination_loss_overtime.npy')
# steps = data['end_steps']
# rewards = data['rewards']
ax[1][0].plot(data)
ax[1][0].set_xlabel('Step')
ax[1][0].set_ylabel('Reward')
ax[1][0].set_title('Hover OC')
ax[1][0]

plt.savefig('../plots/hover_oc_run_info.png')