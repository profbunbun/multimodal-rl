import gymnasium as gym
import torch as T
import numpy as np
from envs.env import Basic
from core.utils import plotLearning

from DQN.Agent2 import Agent2


EPISODES=1000
SHOW_EVERY=100
STEPS=8000
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


env = Basic("nets/3x3/3x3.net.xml","nets/3x3/3x3.rou.xml")
agent = Agent2(4,3,GAMMA,EPS_START,EPS_END,EPS_DECAY,TAU,LR,BATCH_SIZE)




# agent = Agent(gamma=DISCOUNT, epsilon=epsilon, batch_size=BATCH_SIZE, n_actions=3, eps_end=MIN_EPSILON, input_dims=[4], lr=LEARNING_RATE,eps_dec=EPSILON_DECAY)
scores,eps_history=[],[]

count=0
Failed=False
for episode in range(EPISODES):
    score=0
    episode_reward = 0
    
    done=False
    count+=1
    if episode % SHOW_EVERY==0:
        env.render()
    state ,no_choice,lane, out_dict= env.reset()
    start_lane=lane
    step=0
    
    
    
    
    while not done:
         if step >= STEPS:
            done = True
            reward = -1
            score += reward 
            Failed = True  
            print("Starting From: "+start_lane) 
            print("failed after "+str(step) +" steps")
             
         else:
             Failed = False
             lane=lane.partition('_')[0]
             if not no_choice and  lane in out_dict.keys(): 
                temp=list(out_dict[lane].keys())
        
                space=len(out_dict[lane])
                score+=-0.01
                # action = agent.choose_action(state,space)
                action,epsilon=agent.select_action(state,step,episode)
               
                new_state, reward, done, info, no_choice ,lane= env.step(action) 
                score += reward  
                # agent.store_transition(state, action, reward, new_state, done)
                agent.optimize_model()
                state = new_state
                step += 1
                
             else:
                 score+=-0.01
                 action = None
                 new_state, reward, done, info, no_choice ,lane= env.step(action)     
                 score += reward
                 state = new_state
                 step += 1
                
       
    if not Failed:
        print("Starting From: "+start_lane) 
        print("success after "+str(step) +" steps")       
    scores.append(score)
    eps_history.append(epsilon)
    avg_score = np.mean(scores[-100:])
    print('episode: ', episode,'score: %.2f' % score,
            ' average score %.2f' % avg_score,
            'epsilon %.2f' % epsilon)
    x = [i+1 for i in range(len(scores))]
    filename = 'sumo-agent.png'
    plotLearning(x, scores, eps_history, filename)
    
    
        
        
        
    env.close()