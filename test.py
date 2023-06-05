import gymnasium as gym

from envs.env import Basic

# from DQN.DQNAgent import DQNAgent

EPISODES=10
SHOW_EVERY=5
STEPS=2000
DISCOUNT = 0.99
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001


env = Basic("nets/3x3/3x3.net.xml","nets/3x3/3x3.rou.xml")
# agent = DQNAgent()

count=0
for episode in range(EPISODES):
    
    episode_reward = 0
    step = 1
    
    count+=1
    if episode % SHOW_EVERY==0:
        env.render()
    state = env.reset()
    
    
    for  steps in range(STEPS):
        
        state=env.step(0)
        
    env.close()