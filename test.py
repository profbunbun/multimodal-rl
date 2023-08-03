import torch as T
import numpy as np
from Env.env import Basic

from Connector.utility import Utility
from Agent.agent import Agent


EPISODES=1000
STEPS=3000
BATCH_SIZE=32
SUMOCONFIG="Nets/3x3.sumocfg"
env = Basic(SUMOCONFIG,"Nets/3x3b.net.xml","Nets/3x3_2.rou.xml",False)
agent = Agent(4,3)
util=Utility()

rewards,eps_history=[],[]
for episode in range(EPISODES):
    DONE=False
    state ,reward,no_choice,lane, out_dict= env.reset()
    state=T.from_numpy(state)
    STEP=0
    AGENT_STEP=0
    EPISODE_REWARD=0
    
    # fix render
    # env.render('human')
    while not DONE:
             
        if not env.no_choice:
            action=agent.act(state)
            next_state,new_reward, DONE = env.step(action) 
            next_state,new_reward=T.from_numpy(next_state),T.from_numpy(new_reward)
            AGENT_STEP+=1
            agent.remember(state,action,new_reward,next_state,DONE)
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