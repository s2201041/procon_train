import gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import random


import procon_kyougi.kyougi as kyougi
import procon_kyougi.environment as Env 

import numpy as np 

building = [
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0],
    [0,0,1,1,0,0,1,0,0,0,0],
    [0,1,0,0,0,1,0,0,0,0,0],
    [0,0,1,0,2,1,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
]

worker = [
    [[3,3],True],
    [[3,7],True],
    [[6,2],True],

    [[3,4],False],
    [[3,8],False],
    [[6,3],False],
]

log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

env = Env.MyEnv(11,11,building,worker)
env = Monitor(env, log_dir, allow_early_resets=True)

model = PPO('MlpPolicy', env, verbose=0)

for i in range(1000):
    # 学習の実行
    model.learn(total_timesteps=1000)
    model.save("./logs/MyEnv"+str(i))

vec_env = model.get_env()
obs = vec_env.reset()

for i in range(600):
    action, _state = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()