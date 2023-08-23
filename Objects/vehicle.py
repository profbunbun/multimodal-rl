
import numpy as np
from Connector.utility import Utility


import random

util=Utility()


STRAIGHT = "s"
TURN_AROUND = "t"
LEFT = "l"
RIGHT = "r"

# out_mask=[]


class Vehicle:
    
    def __init__(self,vehicle_id,out_dict,index_dict,sumo) -> None:
        
        self.direction_choices = [STRAIGHT, TURN_AROUND,  RIGHT,LEFT]
        self.vehicle_id =vehicle_id
        self.out_dict=out_dict
        self.index_dict=index_dict
        self.sumo = sumo
        self.current_lane=self.sumo.vehicle.getLaneID( self.vehicle_id)
        self.cur_loc=self.current_lane.partition("_")[0]
        
        # if ':' not in self.cur_loc:
        #     self.outlist=list(self.out_dict[self.cur_loc].keys())
        #     self.outlane=list(self.out_dict[self.cur_loc].values())
        
        pass
    
    def get_lane(self):        
        self.current_lane=self.sumo.vehicle.getLaneID( self.vehicle_id)
        self.cur_loc=self.current_lane.partition("_")[0]
        return self.cur_loc
    
    
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
    
    # def get_stats(self):
    #     current_lane=self.get_lane()
    #     current_location=current_lane.partition("_")[0] 
    #     if ':' not in current_lane :     
    #         return      
   
            
        
        
        return 
            
     
    def get_out_dict(self):
        lane = self.get_lane()
        options=self.out_dict[lane]
        return options
    
    
        
    def set_destination(self,action):
        # print(action)
        
        self.current_lane=self.sumo.vehicle.getLaneID( self.vehicle_id)
        self.cur_loc=self.current_lane.partition("_")[0]
        
        if ':' not in self.cur_loc:
            
            outlist=list(self.out_dict[self.cur_loc].keys())
            outlane=list(self.out_dict[self.cur_loc].values())
            out_choices=list(self.out_dict[self.cur_loc])
            # print(out_choices)
            # print(outlist[action],outlane[action])
            
            
            
            if action in outlist : 
                
                target_lane = self.out_dict[self.cur_loc][action]
                # if isinstance(target,str): 
                self.sumo.vehicle.changeTarget( self.vehicle_id,target_lane)
                # else:
                #     self.sumo.vehicle.changeTarget( self.vehicle_id,target[0])
                    
            
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
         self.sumo.vehicle.changeTarget( self.vehicle_id,edgeID=self.new_lane)
         self.sumo.vehicle.moveTo( self.vehicle_id,self.new_lane+"_0",5)
         return
    
    
    