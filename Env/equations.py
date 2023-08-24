"""module stuff"""
import xml.etree.ElementTree as ET
import sumolib
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class MakeGraph:
    """
     _summary_

    _extended_summary_
    """

    def __init__(self, sumocfg) -> None:
        self.sumocfg = sumocfg

    def _parse_net_files(self):
        """
        _parse_net_files _summary_

        _extended_summary_

        Returns:
            _description_
        """
        tree = ET.parse(self.sumocfg)
        root = tree.getroot()
        for infile in root.findall("input"):
            for network in infile.findall("net-file"):
                network_file = str(network.get("value"))
            return network_file

    def get_edges_info(self):
        """
        get_edges_info _summary_

        _extended_summary_

        Returns:
            _description_
        """
        net_file = self._parse_net_files()
        path_ = self.sumocfg.rsplit("/")
        net = sumolib.net.readNet(path_[0] + "/" + net_file)
        out_dict = {}
        length_dict = {}
        index_dict = {}
        edge_list = []
        counter = 0
        lane_pos = {}
        all_edges = net.getEdges()

        for current_edge in all_edges:
            current_edge_id = current_edge.getID()
            edge_start, edge_end = current_edge.getShape()
            x_1 = (edge_start[0], edge_end[0])
            x = np.round(np.mean(x_1), 1)
            y_1 = (edge_start[1], edge_end[1])
            y = np.round(np.mean(y_1), 1)
            print(x, y)
            lane_pos[current_edge_id] = x, y

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

        return [out_dict, lane_pos]

    def edges_nodes(self):
        """
        edges_nodes _summary_

        _extended_summary_

        Returns:
            _description_
        """
        out_dict, lane_pos = self.get_edges_info()
        graph = nx.Graph()
        print(lane_pos)

        for key in out_dict.items():
            # print(key,out_dict[key])
            for edge in out_dict[key]:
                # print (key,out_dict[key][edge])
                graph.add_edge(key, out_dict[key][edge])

        return graph


graph_maker = MakeGraph("Nets/3x3b.sumocfg")

g = graph_maker.edges_nodes()
# print(g)
nx.draw(g)
plt.show()
# print(out_dict.values())
