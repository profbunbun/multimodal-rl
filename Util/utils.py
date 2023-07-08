import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym 

def plotLearning(x, scores, epsilons, filename,exploit_count, explore_count,lines=None):
    count_labels=["exploit","explore"]  
    counts=[exploit_count,explore_count]
    # explore_retio=exploit_count/explore_count
    # exploit_ratio=explore_count/exploit_count
    # fig=plt.figure()
    # ax=fig.add_subplot(111, label="1")
    # ax2=fig.add_subplot(111, label="2", frame_on=False)
    
    
    
    
    
    
    
    
    

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
   
    # ax2.plot(x,exploit_count, color="C2",label="exploit")
    # ax2.set_ylabel("Exploit",color="C2")

    # ax3=ax2.twinx()
    # ax3.plot(x,explore_count, color="C3",label="explore")
    # ax3.set_ylabel("Explore",color="C3")
    # ax3.legend(loc="upper right")
    
    
    
    # axs[1].plot(x,exploit_count, color="C2",label="exploit")
    # axs[1].set_ylabel("Exploit",color="C2")
    # ax3=axs[1].twinx()
    # ax3.plot(x,explore_retio, color="C3",label="explore")
    # ax3.set_ylabel("Exploit",color="C3")

    # axs[1].legend(loc="upper right")
    
    
    
    



   
        
    
   
    
    # axs[0,0].plot(x, epsilons, color="C0")
    # axs[1].scatter(x, running_avg, color="C1")
    
    

    # if lines is not None:
    #     for line in lines:
    #         plt.axvline(x=line)

    plt.savefig(filename)
    plt.close('all')

