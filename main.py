#import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

import time
import random
import os
import numpy as np 

import procon_kyougi.kyougi as kyougi
import procon_kyougi.environment as Env 

# ログフォルダの準備
log_dir = './logs/'
os.makedirs(log_dir, exist_ok=True)

# 学習環境の準備

building = [
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0],
    [0,0,1,1,0,0,1,0],
    [0,1,0,0,0,1,0,0],
    [0,0,1,0,2,1,0,0],
    [0,0,1,0,0,0,0,0],
    [0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0],
]

worker = [
    [[1,1],True],
    [[3,7],True],
    [[4,2],True],
    [[5,4],True],
]

env = Env.MyEnv(9,8,building,worker)

model = A2C('MlpPolicy', env, verbose=1,tensorboard_log=log_dir)

# 学習の実行
model.learn(total_timesteps=10000)

vec_env = model.get_env()

obs = vec_env.reset()
for i in range(30):
    action, _state = model.predict(obs)
    obs, reward, done, info = vec_env.step(action)
    #vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()