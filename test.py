import gymnasium as gym

from envs.env import Basic


EPISODES=10
SHOW_EVERY=5
STEPS=2000

env = Basic("nets/3x3/3x3.net.xml","nets/3x3/3x3.rou.xml")


count=0
for episode in range(EPISODES):
    count+=1
    if episode % SHOW_EVERY==0:
        env.render()
    env.reset()
    for  steps in range(STEPS):
        env.step()
    env.close()