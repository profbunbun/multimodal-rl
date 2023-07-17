import os
import sys
import matplotlib.pyplot as plt


import xml.dom.minidom
from xml.dom.minidom import parse, parseString
import numpy as np
import pandas as pd

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")

import sumolib
from sumolib import checkBinary

class Utility:
    def __init__(self) -> None:
        pass
    
    def getNetInfo(self,net_file_name):
   
        if net_file_name.endswith('.net.xml'):
            return sumolib.net.readNet(net_file_name)
        return None
    
    def getEdgesInfo(self,net):
    
        out_dict = {}
        length_dict = {}
        index_dict = {}
        edge_list = []
        counter = 0
        all_edges = net.getEdges()
        #all_connections = net.getConnections()
        for current_edge in all_edges:
            current_edge_id = current_edge.getID()
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

        return [ out_dict, index_dict, edge_list] 
    
    def plotLearning(self,x, scores, epsilons, filename,lines=None):
        N = len(scores)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])
            
        
        fig,ax1=plt.subplots(1,figsize=(10,10))
        ax1.plot(x, running_avg, color="C1" ,label="Reward")
        ax1.set_ylabel("Reward",color="C1")
        ax1.legend(loc="upper left")
        axa=ax1.twinx()
        axa.plot(x, epsilons, color="C0",label="epsilon")
        axa.set_ylabel("Epsilon",color="C0")
        axa.legend(loc="upper right")
    
        plt.savefig(filename)
        plt.close('all')
        pass
    
    def getMinMax(self,infile):
        file= infile

        doc = parse(file)

        root = doc.documentElement
            
        edges= root.getElementsByTagName("edge")

        index = 0
        array= np.array([])
        for edge in edges:
            lanes = edge.getElementsByTagName("lane")
            
            for lane in lanes:
                    shape=lane.getAttribute("shape")
                    shape1=shape.split(" ")
                    array= np.append([array],[shape1])
                    
        array2= np.array([])                     
        for element in range(len(array)):
                str1=str(array[element])
                
                str2=str1.split(",")
            
                array2=np.append([array2],[str2])

        array3=np.array([])
        for element in range(len(array2)):
                np.append([array3],[array2[element]])

        df = pd.DataFrame(array2)
        df=pd.to_numeric(df[0],downcast="float")
        max=df.max()
        min=df.min()
        diff=max-min
    
        
        return min,max,diff
    
    def translate(self,value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
        valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
        return int(round((rightMin + (valueScaled * rightSpan)),0))
