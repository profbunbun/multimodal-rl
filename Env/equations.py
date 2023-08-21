import sumolib
from xml.dom.minidom import parse
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
    
    
    
    
class Make_Graph:
    def __init__(self,sumocfg) -> None:
        self.sumocfg=sumocfg
    
    def _parse_net_files(self):
        tree = ET.parse(self.sumocfg)
        root = tree.getroot()
        for input in root.findall('input'):            
            for network in input.findall('net-file'):
                network_file=str(network.get('value'))
            return network_file

    def getEdgesInfo(self):
        net_file=self._parse_net_files()
        path_=self.sumocfg.rsplit('/')
        net= sumolib.net.readNet(path_[0]+"/"+net_file)
        out_dict = {}
        length_dict = {}
        index_dict = {}
        edge_list = []
        counter = 0
        lane_pos={}
        all_edges = net.getEdges()
        
        for current_edge in all_edges:
            current_edge_id = current_edge.getID()
            edge_start,edge_end=current_edge.getShape()
            x1=((edge_start[0],edge_end[0]))
            x = np.round(np.mean(x1),1)
            y1=((edge_start[1],edge_end[1]))
            y = np.round(np.mean(y1),1)
            print(x,y)
            lane_pos[current_edge_id]= x,y
            
            if current_edge.allows("passenger"):
                edge_list.append(current_edge)
            if current_edge_id in index_dict.keys():
                print(current_edge_id+" already exists!")
            else:
                index_dict[current_edge_id] = counter
                counter += 1
            if current_edge_id in out_dict.keys():
                print(current_edge_id+" already exists!")
            else:
                out_dict[current_edge_id] = {}
            if current_edge_id in length_dict.keys():
                print(current_edge_id+" already exists!")
            else:
                length_dict[current_edge_id] = current_edge.getLength()
            #edge_now is sumolib.net.edge.Edge
            out_edges = current_edge.getOutgoing()
            for current_out_edge in out_edges:
                if not current_out_edge.allows("passenger"):
                    #print("Found some roads prohibited")
                    continue
                conns = current_edge.getConnections(current_out_edge)
                for conn in conns:
                    dir_now = conn.getDirection()
                    out_dict[current_edge_id][dir_now] = current_out_edge.getID()
        
        # all_nodes= net.getNodes()
        # for current_node in all_nodes:
        #     current_node_id = current_node.getID()
        #     # dir(current_node_id)
        #     position= net.getNode(current_node_id).getCoord()

            # print(position)
        
        return [ out_dict, index_dict, edge_list, lane_pos]
    
    
    def edges_nodes(self):
        out_dict, index_dict, edge_list ,lane_pos=self.getEdgesInfo()
        graph=nx.Graph()
        print(lane_pos)
        # for edge_1 in edge_list:
        #     # print(edge_1._id,edge_1._from._id,edge_1._to._id)
        #     for edge_2 in edge_list:
        #         if edge_1._from._id == edge_2._to._id:
        #             graph.append([edge_1._id,edge_2._id])
                    
        # pr
        # print(index_dict)
        
        for key in out_dict:
            # print(key,out_dict[key])
            for edge in out_dict[key]:
                # print (key,out_dict[key][edge])
                graph.add_edge(key,out_dict[key][edge])
        
        
        return graph
    
graph_maker=Make_Graph("Nets/3x3b.sumocfg")

g = graph_maker.edges_nodes()
# print(g)
nx.draw(g)
plt.show()
# print(out_dict.values())