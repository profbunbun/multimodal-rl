import os
import sys
import numpy as np
from Util.utility import Utility
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")
import sumolib
from sumolib import net
import traci

import random
LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ
util=Utility()



class Vehicle:
    CONNECTION_LABEL = 0 
    def __init__(self,vehicle_id,net_file,route_file,out_dict,index_dict,sumo) -> None:
        
        self.vehicle_id =vehicle_id
        self.destination = None
        self._net = net_file
        self._route = route_file
        self.min=0
        self.max=0
        self.diff=0
        self.out_dict=out_dict
        self.index_dict=index_dict
        self.choice=None
        self.flag=None
        self.choice_dic=None
        self.make_choice=True
        
        
        self.sumo = sumo
         
        self.min,self.max,self.diff=util.getMinMax(self._net)
        pass
    
    def random_relocate(self):
         self.new_lane=random.choice(list(self.index_dict.keys()))      
         self.sumo.vehicle.changeTarget("1",edgeID=self.new_lane)
         self.sumo.vehicle.moveTo("1",self.new_lane+"_0",5)
         return
     
     
    def get_start_lane(self):
        return self.new_lane
    
    def location(self):
        
        self.vpos=self.sumo.vehicle.getPosition(self.vehicle_id)
        
        
        return [self.vpos[0],self.vpos[1]] 
        
    def set_destination(self,action):
        # print(action)
        
        self.current_lane=self.sumo.vehicle.getLaneID("1")
        self.cur_loc=self.current_lane.partition("_")[0]
        
        if ':' not in self.cur_loc:
            
            outlist=list(self.out_dict[self.cur_loc].keys())
            outlane=list(self.out_dict[self.cur_loc].values())
            if action < len(outlist) : 
                outlist=np.array(outlist)
                outlane=np.array(outlane)
                target = outlane[action]
                
                self.sumo.vehicle.changeTarget("1",target)
        return

    def get_out_dict(self):
        return self.out_dict
    
    def links(self):
        
        current_lane=self.sumo.vehicle.getLaneID("1")
        choices=len(self.sumo.lane.getLinks(current_lane))
        return choices
    
    def get_lane(self):
        
        current_lane=self.sumo.vehicle.getLaneID("1")
        return current_lane
    
    def set_pickup(self):
        pass
    
    def pickup(self):
        reservation=self.sumo.person.getTaxiReservations(0)
        reservation_id=reservation[0]
        # self.sumo.vehicle.dispatchTaxi("1","0")
        # print(reservation_id)
        pass
    def close(self):
        self.sumo.close()
        return
    
    
    
    