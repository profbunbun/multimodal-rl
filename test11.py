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
rewards,eps_history=[],[]   
for episode in range(EPISODES):
    done=False
    state ,reward,no_choice,lane, out_dict= env.reset()
    step=0
    while not done:
             
             if not env.no_choice:
                action=agent.act(state)
             else:
                 action=-1   
            #  print(state)
            #  print(action)
             
             next_state,new_reward, done = env.step(action) 
             step+=1
            #  print(next_state)
             agent.remember(state,action,new_reward,next_state,done)
             state=next_state
             if len(agent.memory)> batch_size:
                        agent.replay(batch_size)
    
    
    agent.epsilon_decay()         
    r = float(new_reward)   
    rewards.append(r)
    eps_history.append(agent.epsilon)
    avg_reward = np.mean(rewards[-100:])         
    print('---------episode: ', episode,'reward: %.2f' % r,
            ' average reward %.2f' % avg_reward  ,
            'epsilon %.2f' % agent.epsilon," **** step: ",step)
    x = [i+1 for i in range(len(rewards))]
    filename = 'sumo-agent.png'
    plotLearning(x, rewards, eps_history, filename)              
    env.close()