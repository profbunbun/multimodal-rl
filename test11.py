import gymnasium as gym
import torch as T
import numpy as np
from Env.env import Basic

from Util.utils import plotLearning

from Agent.agent6 import Agent6


EPISODES=100
STEPS=5000
batch_size=32
env = Basic("Nets/3x3.net.xml","Nets/S3x3.rou.xml",STEPS)
agent = Agent6(4,3)

for episode in range(EPISODES):
    done=False
    state ,reward,no_choice,lane, out_dict= env.reset()
    
    while not done:
             
             action=agent.act(state)
             next_state,reward, done = env.step(action) 
             agent.remember(state,action,reward,next_state,done)
             
    if len(agent.memory)> batch_size:
                 agent.replay(batch_size)
                  
    env.close()