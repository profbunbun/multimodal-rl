import gymnasium as gym
import numpy as np
from envs.env import Basic
from core.utils import plotLearning
from DQN.DQNAgent import Agent

EPISODES=1000
SHOW_EVERY=100
STEPS=1000
DISCOUNT = 0.99
epsilon = 1.0  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.01


env = Basic("nets/3x3/3x3.net.xml","nets/3x3/3x3.rou.xml")
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=3, eps_end=0.01, input_dims=[4], lr=0.003)
scores,eps_history=[],[]

count=0
Failed=False
for episode in range(EPISODES):
    score=0
    episode_reward = 0
    
    done=False
    count+=1
    if epsilon <= 0.2:
        agent.epsilon = 1.0
    if episode % SHOW_EVERY==0:
        env.render()
    state ,no_choice,start_lane, out_dict= env.reset()
    lane=start_lane
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
                print("Temp "+str(temp))
                print("LANE: "+lane,"OUT_DICT: "+str(out_dict[lane]))
                space=len(out_dict[lane])
                score+=-0.01
                action = agent.choose_action(state,space)
                # print("Out_dict: "+str(out_dict[lane][action])) 
                new_state, reward, done, info, no_choice ,lane= env.step(action) 
                score += reward  
                agent.store_transition(state, action, reward, new_state, done)
                agent.learn()
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
    eps_history.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])
    print('episode: ', episode,'score: %.2f' % score,
            ' average score %.2f' % avg_score,
            'epsilon %.2f' % agent.epsilon)
    x = [i+1 for i in range(len(scores))]
    filename = 'sumo-agent.png'
    plotLearning(x, scores, eps_history, filename)
    
    
        
        
        
    env.close()