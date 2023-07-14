import gymnasium as gym
import torch as T
import numpy as np
from Env.env2 import Basic

from Util.utils import plotLearning

from Agent.agent9 import Agent9


EPISODES=10000
STEPS=3000
batch_size=32
env = Basic("Nets/3x3.net.xml","Nets/S3x3.rou.xml",False,STEPS)
agent = Agent9(4,3)
rewards,eps_history=[],[]   
for episode in range(EPISODES):
    done=False
    state ,reward,no_choice,lane, out_dict= env.reset()
    state=T.from_numpy(state)
    step=0
    agent_step=0
    while not done:
             
             if not env.no_choice:
                action=agent.act(state)
                next_state,new_reward, done = env.step(action) 
                next_state,new_reward=T.from_numpy(next_state),T.from_numpy(new_reward)
                agent_step+=1
                agent.remember(state,action,new_reward,next_state,done)
                state=next_state
             else:
                 env.nullstep()
            
             if (len(agent.memory)> batch_size) and (step % batch_size == 0):
                        agent.replay(batch_size)
             step+=1
    
    agent.epsilon_decay_2(episode,EPISODES)         
    r = float(new_reward)   
    rewards.append(r)
    eps_history.append(agent.epsilon)
    avg_reward = np.mean(rewards[-100:])         
    print('---------episode: ', episode,'reward: %.3f' % r,
            ' average reward %.3f' % avg_reward  ,
            'epsilon %.5f' % agent.epsilon," **** step: ",step,"*** Agent steps: ", agent_step)
    x = [i+1 for i in range(len(rewards))]
    filename = 'sumo-agent.png'
    plotLearning(x, rewards, eps_history, filename)              
    env.close()