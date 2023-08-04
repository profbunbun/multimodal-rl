import math
import numpy as np
from Objects.vehicle import Vehicle
from Objects.person import Person
from Connector.connect import SUMOConnection

class Basic():

    def __init__(self,sumocon:str) -> None:
        
        self.sumo_con=SUMOConnection(sumocon)
        self.out_dict, self.index_dict,self.edge_list = self.sumo_con.getEdgesInfo()
        
        
        self.episode_count = 0
        self.episode_step = 0
        self.action = 0

        self.done = False
        self.no_choice = False

        self.vehicle = None

       
        

    def reset(self):
        self.sumo.simulationStep()
        self.reward = 0
        self.episode_step += 1
        self.done = False
        self.steps = 0
        self.agent_step = 0

        self.vehicle = Vehicle("1", self.out_dict, self.index_dict, self.sumo)
        
        out_mask= self.vehicle.get_stats()
        self.vedge = self.sumo.vehicle.getRoadID("1")
        
        self.vloc = self.vehicle.location()
            
            
        self.person = Person("p_0",  self.sumo)
        self.pedge = self.sumo.person.getRoadID("p_0")
       
        self.ploc = self.person.location()



        self.new_distance = (math.dist(self.vloc, self.ploc))

        state = np.array([])
        state = np.append(state, self.vloc)
        state = np.append(state, self.ploc)
        state = np.append(state, self.steps)
        state = np.append(state, self.new_distance)

        self.done = False
        self.no_choice = False

        self.lane = self.sumo.vehicle.getLaneID("1")
        self.out_dict = self.vehicle.get_out_dict()
        
        
            
        return state, self.reward,  self.done, out_mask
    
    def nullstep(self):
        self.reward = 0
        self.old_edge = self.vedge
        self.sumo.simulationStep()
        self.vedge = self.sumo.vehicle.getRoadID("1")
        
        if  ':' not in self.vedge and self.old_edge == self.vedge:
            self.no_choice = False
        else:
            self.no_choice = True 
        pass

    def step(self, action=None):
        self.reward = 0
        self.old_distance = self.new_distance

        self.steps += 1

        self.action = action
        self.old_edge = self.vedge
        self.sumo.simulationStep()
        
       
       
        self.vedge = self.sumo.vehicle.getRoadID("1")
        
        out_mask= self.vehicle.get_stats()
            

        if  ':' not in self.vedge and self.old_edge == self.vedge:
            self.no_choice = False
        else:
            self.no_choice = True

        self.lane = self.sumo.vehicle.getLaneID("1")
       
        self.vloc = self.vehicle.location()
        self.ploc = self.person.location()
        self.new_distance = (math.dist(self.vloc, self.ploc))

        
        if not self.no_choice:
           
            if self.new_distance > self.old_distance:
                self.reward += -.2
            if self.new_distance < self.old_distance:
                self.reward += .15
            
            self.vehicle.set_destination(action)
            self.reward += -.1
            self.agent_step += 1

        self.vedge = self.sumo.vehicle.getRoadID("1")
        self.pedge = self.sumo.person.getRoadID("p_0")

        if ":" not in self.vedge:
            self.vehicle_lane_index = self.index_dict[self.vedge]
        else:
            self.vehicle_lane_index = self.vehicle_lane_index

        self.person_lane_index = self.index_dict[self.pedge]

        done = False

        if self.vedge == self.pedge:
            self.reward += 5
            done = True
        
            

        state = np.array([])
        state = np.append(state, self.vloc)
        state = np.append(state, self.ploc)
        # state = np.append(state, self.vehicle_lane_index)
        # state = np.append(state, self.person_lane_index)
        state = np.append(state, self.agent_step)
        state = np.append(state, self.new_distance)

        reward = np.array([])
        reward = np.append(reward, self.reward)

        return state, reward, done, out_mask

    def render(self, mode):
        
        if mode =="gui":
            self.sumo = self.sumo_con.connect_gui()
        
        elif mode =="libsumo":
            self.sumo = self.sumo_con.connect_libsumo_no_gui()
        
        elif mode == "no_gui":
            self.sumo = self.sumo_con.connect_no_gui()
        
        
        return

    def close(self):
        
        self.sumo.close()
        return
