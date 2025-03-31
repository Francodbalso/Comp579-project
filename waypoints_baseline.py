import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import time
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv
import numpy as np
import matplotlib.pyplot as plt 

env = gym.make('PyFlyt/QuadX-Waypoints-v4')
env = FlattenWaypointEnv(env, context_length=2)
test_env = gym.make('PyFlyt/QuadX-Waypoints-v4')
test_env = FlattenWaypointEnv(test_env, context_length=2)

model = PPO("MlpPolicy", env, verbose=0)
n_timesteps = int(4e5)

eval_callback = EvalCallback(test_env,
                             log_path='../data/waypoint_baseline/',
                             eval_freq=5000,
                             n_eval_episodes=2)

t = time.time()
model.learn(total_timesteps=n_timesteps, callback=eval_callback)
print('time taken: ', time.time() - t)