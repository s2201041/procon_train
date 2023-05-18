import gym
import time
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import procon_kyougi.kyougi as kyougi
import procon_kyougi.environment as Env 

building = [
    [0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0,0],
    [0,0,1,1,0,0,1,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,2,0,0,0,1,0,0],
    [0,0,1,0,0,0,0,1,1,0,0],
    [0,0,1,0,0,0,1,0,1,0,0],
    [0,0,0,0,1,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,1,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0],
]

worker = [
    [[3,3],True],
    [[3,8],True],
    [[9,7],True],

    [[3,4],False],
    [[3,8],False],
    [[6,3],False],
]

env = Env.MyEnv(11,11,building,worker)

model = PPO('MlpPolicy', env, verbose=0)
model = model.load("./logs/MyEnv"+str(43))

obs = env.reset()
env.render("human")

for i in range(30):
    action, _state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render("human")
    time.sleep(0.3)
    # VecEnv resets automatically
    #if done:
    #   obs = env.reset()
