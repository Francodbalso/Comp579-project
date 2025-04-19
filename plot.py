import numpy as np
import matplotlib.pyplot as plt

# path = '../data/hover_oc/test_run.npz'
# savename = 'hover_oc_run_info'
path = '../data/pendulum/test_run.npz'
savename = 'pendulum_oc'

fig, ax = plt.subplots(2, 2, figsize=(12,8))

data = np.load(path)
end_steps = data['end_steps']
update_steps = data['update_steps']
rewards = data['rewards']
pol_loss = data['pol_loss']
q_loss = data['q_loss']
term_loss = data['term_loss']

ax[0][0].plot(end_steps, rewards)
ax[0][0].set_xlabel('Step')
ax[0][0].set_ylabel('Reward')
ax[0][0].set_title('Hover OC Reward')

ax[0][1].plot(update_steps, pol_loss)
ax[0][1].set_xlabel('Step')
ax[0][1].set_ylabel('Loss')
ax[0][1].set_title('Policy Loss')

ax[1][1].plot(update_steps, q_loss)
ax[1][1].set_xlabel('step')
ax[1][1].set_ylabel('Loss')
ax[1][1].set_title('Q loss')

ax[1][0].plot(update_steps, term_loss)
ax[1][0].set_xlabel('Step')
ax[1][0].set_ylabel('Reward')
ax[1][0].set_title('Term loss')

fig.tight_layout()
plt.savefig(f'../plots/{savename}.png')
