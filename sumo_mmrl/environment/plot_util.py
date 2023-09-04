"""module stuff """
import matplotlib.pyplot as plt
import numpy as np



class Utility:
    """
     _summary_

    _extended_summary_
    """

    def __init__(self) -> None:
        pass


    def plot_learning(self, x, scores, epsilons, filename):
        """
        plotLearning _summary_

        _extended_summary_

        Arguments:
            x -- _description_
            scores -- _description_
            epsilons -- _description_
            filename -- _description_

        Keyword Arguments:
            lines -- _description_ (default: {None})
        """
        number_of_scores = len(scores)
        running_avg = np.empty(number_of_scores)
        # avg_score = np.mean(scores)
        for t in range(number_of_scores):
            running_avg[t] = np.mean(scores[max(0, t - 20) : (t + 1)])

        ax1 = plt.subplot(111)
        ax1.plot(x, running_avg, color="C1", label="Reward")
        # ax1.plot(x, running_avg, color="C1" ,label="Reward")
        ax1.set_ylabel("Reward", color="C1")
        ax1.legend(loc="upper left")
        axa = ax1.twinx()
        axa.plot(x, epsilons, color="C0", label="epsilon")
        axa.set_ylabel("Epsilon", color="C0")
        axa.legend(loc="upper right")

        plt.savefig(filename)
        plt.close("all")
