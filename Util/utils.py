import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym 

def plotLearning(x, scores, epsilons, filename,lines=None):
    
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])
        
    
    fig,ax1=plt.subplots(1,figsize=(10,10))
    ax1.plot(x, running_avg, color="C1" ,label="score")
    ax1.set_ylabel("Score",color="C1")
    ax1.legend(loc="upper left")
    axa=ax1.twinx()
    axa.plot(x, epsilons, color="C0",label="epsilon")
    axa.set_ylabel("Epsilon",color="C0")
    axa.legend(loc="upper right")
   
    plt.savefig(filename)
    plt.close('all')

