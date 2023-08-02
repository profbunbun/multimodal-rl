import os
import sys
import numpy as np
from Util.utility import Utility


import random

util=Utility()



class Vehicle:
    
    def __init__(self,vehicle_id,net_file,route_file,out_dict,index_dict,sumo) -> None:
        self.vehicle_id =vehicle_id
        self._net = net_file
        self._route = route_file
        self.out_dict=out_dict
        self.index_dict=index_dict
        self.sumo = sumo
        self.current_lane=self.sumo.vehicle.getLaneID("1")
        self.cur_loc=self.current_lane.partition("_")[0]
        if ':' not in self.cur_loc:
            
            self.outlist=list(self.out_dict[self.cur_loc].keys())
            self.outlane=list(self.out_dict[self.cur_loc].values())
        
        pass
    
    def get_lane(self):        
        current_lane=self.sumo.vehicle.getLaneID("1")
        return current_lane
    
    
    def location(self):
        self.vpos=self.sumo.vehicle.getPosition(self.vehicle_id)
        return [self.vpos[0],self.vpos[1]] 
    
    def is_indexed_lane(self):
        current_lane=self.get_lane()
        if current_lane == '':
            return False
        
        if ':' not in current_lane :
            return True
        else:
            return False
    
    def get_stats(self):
        current_lane=self.get_lane()
        current_location=current_lane.partition("_")[0]            
        out_choices=list(self.out_dict[current_location].keys())
        out_lanes=list(self.out_dict[current_location].values())
        return current_lane,out_choices,out_lanes
            
     
    def get_out_dict(self):
        return self.out_dict
    
    
        
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
    
    def random_relocate(self):
         self.new_lane=random.choice(list(self.index_dict.keys()))      
         self.sumo.vehicle.changeTarget("1",edgeID=self.new_lane)
         self.sumo.vehicle.moveTo("1",self.new_lane+"_0",5)
         return
    
    
    