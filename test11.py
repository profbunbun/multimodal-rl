import gymnasium as gym
import torch as T
import numpy as np
from envs.env import Basic

from core.utils import plotLearning

from DQN.Agent5 import Agent5


EPISODES=5000
SHOW_EVERY=100
STEPS=5000
BATCH_SIZE = 64
GAMMA = 0.9997
epsilon=0.98
EPS_MAX = 0.98
EPS_END = 0.1
EPS_DECAY = 500 
TAU = 0.005
LR = 1e-3


env = Basic("nets/3x3/3x3.net.xml","nets/3x3/3x3.rou.xml",STEPS)
agent = Agent5(4,3,GAMMA,epsilon,EPS_MAX,EPS_END,EPS_DECAY,TAU,LR,BATCH_SIZE)

scores,eps_history=[],[]

count=0
Failed=False
for episode in range(EPISODES):
    score=0
    episode_reward = 0
    
    done=False
    count+=1
    # if episode % SHOW_EVERY==0:
    #     env.render()
    state ,reward,no_choice,lane, out_dict= env.reset()
    previous_action=0
    agent.exploit_count=0
    agent.explore_count=0
    
    start_lane=lane
    step=0
    
    
    
    
    while not done:
         
            
             
        
             Failed = False
             lane=lane.partition('_')[0]
             if not no_choice and  lane in out_dict.keys(): 
                
                # action = agent.choose_action(state,space)
                action,epsilon,explore_count,exploit_count=agent.select_action(state,step,episode,EPISODES)
                previous_action=action.detach().cpu().numpy()
                action=action.detach().cpu().numpy()
                new_state,reward, done, info, no_choice ,lane= env.step(action) 
                # reward=new_state[2]
                # print(reward)
                score = reward
                state = new_state
                # print(action)
                action=action[0][0]
                this_action=np.array([])
                this_action=np.append(this_action,action)
                this_action=T.from_numpy(this_action)
                
                agent.memory.push(state, this_action,  new_state,reward)
                agent.optimize_model()
                # print(state)
                step += 1
                
             else:
                 
                 action = None
                #  action=action.detach().cpu().numpy()
                 new_state,reward, done, info, no_choice ,lane= env.step(action)    
                #  reward=new_state[2]
                #  print(reward) 
                 score = reward
                 state = new_state
                 step += 1
                
       
   
    score = float(score)   
    scores.append(score)
    eps_history.append(epsilon)
    avg_score = np.mean(scores[-100:])
  
    
    print('---------episode: ', episode,'score: %.2f' % score,
            ' average score %.2f' % avg_score,
            'epsilon %.2f' % epsilon," **** step: ",step)
    x = [i+1 for i in range(len(scores))]
    filename = 'sumo-agent.png'
    plotLearning(x, scores, eps_history, filename,explore_count,exploit_count)
    
    
        
        
        
    env.close()