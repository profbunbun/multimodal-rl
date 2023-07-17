import gymnasium as gym
import torch as T
import numpy as np
from Env.env import Basic

from Util.utility import Utility
from Agent.agent import Agent


EPISODES=1000
STEPS=3000
batch_size=32
env = Basic("Nets/3x3.net.xml","Nets/S3x3.rou.xml",False)
agent = Agent(3,3)
util=Utility()

rewards,eps_history=[],[]   
for episode in range(EPISODES):
    done=False
    state ,reward,no_choice,lane, out_dict= env.reset()
    state=T.from_numpy(state)
    step=0
    agent_step=0
    episode_reward=0
    
    # fix render
    # env.render('human')
    while not done:
             
             if not env.no_choice:
                action=agent.act(state)
                next_state,new_reward, done = env.step(action) 
                next_state,new_reward=T.from_numpy(next_state),T.from_numpy(new_reward)
                agent_step+=1
                agent.remember(state,action,new_reward,next_state,done)
                state=next_state
                episode_reward+=new_reward
                # if (len(agent.memory)> 1000) :
                #         agent.replay(batch_size)
                
                
             else:
                 env.nullstep()
            
             
             step+=1
             
    if (len(agent.memory)> batch_size) :
                        agent.replay(batch_size)         
    
    agent.epsilon_decay_2(episode,EPISODES)   
    
          
    r = float(episode_reward)   
    rewards.append(r)
    eps_history.append(agent.epsilon)
    # avg_reward = np.mean(rewards[-100:])
    avg_reward = np.mean(rewards)   
          
    print('EP: ', episode,'Reward: %.3f' % r,
            ' Average Reward %.3f' % avg_reward  ,
            'epsilon %.5f' % agent.epsilon," **** step: ",step,"*** Agent steps: ", agent_step)
    x = [i+1 for i in range(len(rewards))]
    filename = 'sumo-agent.png'
    
    util.plotLearning(x, rewards, eps_history, filename)              
    env.close()