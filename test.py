import gymnasium as gym
import numpy as np
from envs.env import Basic
import matplotlib.pyplot as plt
import pickle
import time


from mpl_toolkits.mplot3d import axes3d

from matplotlib import style

EPISODES=10
SHOW_EVERY=5
STEPS=2000

env = Basic("nets/3x3/3x3.net.xml","nets/3x3/3x3.rou.xml")

        
   

# sf


count=0
for episode in range(EPISODES):
    count+=1
    if episode % SHOW_EVERY==0:
        env.render()
    env.reset()
    for  steps in range(STEPS):
        env.step()
    env.close()