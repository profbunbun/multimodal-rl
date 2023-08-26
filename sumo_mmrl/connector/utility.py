"""module stuff """
from xml.dom.minidom import parse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sumolib


class Utility:
    """
     _summary_

    _extended_summary_
    """

    def __init__(self) -> None:
        pass

    def get_net_info(self, net_file_name):
        """
        getNetInfo _summary_

        _extended_summary_

        Arguments:
            net_file_name -- _description_

        Returns:
            _description_
        """
        if net_file_name.endswith(".net.xml"):
            return sumolib.net.readNet(net_file_name)
        return None

    def get_edges_info(self, net):
        """
        getEdgesInfo _summary_

        _extended_summary_

        Arguments:
            net -- _description_

        Returns:
            _description_
        """

        out_dict = {}
        length_dict = {}
        index_dict = {}
        edge_list = []
        counter = 0
        all_edges = net.getEdges()
        # all_connections = net.getConnections()
        for current_edge in all_edges:
            current_edge_id = current_edge.getID()
            if current_edge.allows("passenger"):
                edge_list.append(current_edge)
            if current_edge_id in index_dict:
                print(current_edge_id + " already exists!")
            else:
                index_dict[current_edge_id] = counter
                counter += 1
            if current_edge_id in out_dict:
                print(current_edge_id + " already exists!")
            else:
                out_dict[current_edge_id] = {}
            if current_edge_id in length_dict:
                print(current_edge_id + " already exists!")
            else:
                length_dict[current_edge_id] = current_edge.getLength()
            # edge_now is sumolib.net.edge.Edge
            out_edges = current_edge.getOutgoing()
            for current_out_edge in out_edges:
                if not current_out_edge.allows("passenger"):
                    # print("Found some roads prohibited")
                    continue
                conns = current_edge.getConnections(current_out_edge)
                for conn in conns:
                    dir_now = conn.getDirection()
                    out_dict[current_edge_id][dir_now] = current_out_edge.getID()

        return [out_dict, index_dict, edge_list]

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

    def get_min_max(self, infile):
        """
        getMinMax _summary_

        _extended_summary_

        Arguments:
            infile -- _description_

        Returns:
            _description_
        """
        file = infile

        doc = parse(file)

        root = doc.documentElement

        edges = root.getElementsByTagName("edge")

        array = np.array([])
        for edge in edges:
            lanes = edge.getElementsByTagName("lane")

            for lane in lanes:
                shape = lane.getAttribute("shape")
                shape1 = shape.split(" ")
                array = np.append([array], [shape1])

        array2 = np.array([])
        for element in array:
            str1 = str(element)

            str2 = str1.split(",")

            array2 = np.append([array2], [str2])

        array3 = np.array([])
        for element in array2:
            np.append([array3], [array2[element]])

        df = pd.DataFrame(array2)
        df = pd.to_numeric(df[0], downcast="float")
        max_df = df.max()
        min_df = df.min()
        diff = max_df - min_df

        return min_df, max_df, diff

    def translate(self, value, left_min, left_max, right_min, right_max):
        """
        translate _summary_

        _extended_summary_

        Arguments:
            value -- _description_
            leftMin -- _description_
            leftMax -- _description_
            rightMin -- _description_
            rightMax -- _description_

        Returns:
            _description_
        """
        # Figure out how 'wide' each range is
        left_span = left_max - left_min
        right_span = right_max - right_min

        # Convert the left range into a 0-1 range (float)
        value_scaled = float(value - left_min) / float(left_span)

        # Convert the 0-1 range into a value in the right range.
        return int(round((right_min + (value_scaled * right_span)), 0))
