import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import time
import PyFlyt.gym_envs
import numpy as np
import matplotlib.pyplot as plt 

env = gym.make('PyFlyt/QuadX-Hover-v4')
test_env = gym.make('PyFlyt/QuadX-Hover-v4')

model = PPO("MlpPolicy", env, verbose=0)
n_timesteps = 60000

eval_callback = EvalCallback(test_env,
                             log_path='../data/',
                             eval_freq=2000,
                             n_eval_episodes=2)

t = time.time()
model.learn(total_timesteps=n_timesteps, callback=eval_callback)
print('time taken: ', time.time() - t)