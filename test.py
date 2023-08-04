import torch as T
import numpy as np
from Env.env import Basic

from Connector.utility import Utility
from Agent.agent import Agent


EPISODES=100
STEPS=3000
BATCH_SIZE=32

SUMOCONFIG="Nets/3x3.sumocfg"
env = Basic(SUMOCONFIG)
agent = Agent(6,3)
util=Utility()

rewards,eps_history=[],[]
for episode in range(EPISODES):
    
    if episode % 10 == 0:
        env.render("gui")
    else:
        env.render("libsumo")
    
    state ,reward,done,out_mask= env.reset()
    state,out_mask=T.from_numpy(state),T.from_numpy(out_mask)
    STEP=0
    AGENT_STEP=0
    EPISODE_REWARD=0
    
    # fix render
    # env.render('human')
    while not done:
             
        if not env.no_choice:
            action=agent.act(state)
            next_state,new_reward, done,out_mask = env.step(action) 
            next_state,new_reward,out_mask=T.from_numpy(next_state),T.from_numpy(new_reward),T.from_numpy(out_mask)
            AGENT_STEP+=1
            agent.remember(state,action,new_reward,next_state,done,out_mask)
            state=next_state
            EPISODE_REWARD+=new_reward
        # if (len(agent.memory)> 1000) :
        #         agent.replay(BATCH_SIZE)
        
        
        else:
            env.nullstep()
    
        if (len(agent.memory)> BATCH_SIZE) and (STEP % BATCH_SIZE == 0):
                    agent.replay(BATCH_SIZE)
                
        STEP+=1
             
    # if (len(agent.memory)> BATCH_SIZE) :
    #                     agent.replay(BATCH_SIZE)         
    
    agent.epsilon_decay_2(episode,EPISODES)   
    
          
    r = float(EPISODE_REWARD)  
    # r = float(new_reward)   
    rewards.append(r)
    
    eps_history.append(agent.epsilon)
    avg_reward = np.mean(rewards[-100:])
    
    # avg_reward = np.mean(rewards)   
       
    print('EP: ', episode,'Reward: %.3f' % r,
            ' Average Reward %.3f' % avg_reward  ,
            'epsilon %.5f' % agent.epsilon," **** STEP: ",STEP,"*** Agent STEPs: ", AGENT_STEP)
    x = [i+1 for i in range(len(rewards))]
    filename = 'sumo-agent.png'
    
    util.plotLearning(x, rewards, eps_history, filename)              
    env.close()