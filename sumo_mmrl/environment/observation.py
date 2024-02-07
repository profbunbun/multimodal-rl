""" Observation Class"""
from .outmask import OutMask


class Observation:
  
    def __init__(self):

        self.out_mask = OutMask()

    
    def get_state(self, sumo, step, vehicle, destination_loc, life, distcheck):
      
        bounding_box = sumo.simulation.getNetBoundary()
        self.min_x, self.min_y = bounding_box[0]
        self.max_x, self.max_y = bounding_box[1]

        # pad max and min with 1 to allow for vehicles to be on the edge
        self.max_manhat_dist = abs((self.max_x + 1) - (self.min_x - 1)) + abs((self.max_y + 1) - (self.min_y - 1))
        
        vloc = vehicle.location()
        choices = vehicle.get_out_dict()
        state = []
        
        state.append(self.normalize(vloc[0], self.min_x, self.max_x))
        state.append(self.normalize(vloc[1], self.min_y, self.max_y))
        state.append(self.normalize(destination_loc[0], self.min_x, self.max_x))
        state.append(self.normalize(destination_loc[1], self.min_y, self.max_y))
        state.append(step * 0.0001)
        state.append(self.manhat_dist(vloc[0], vloc[1], destination_loc[0], destination_loc[1]))  
        state.extend(self.out_mask.get_outmask_valid(choices))
        state.append(life)
        state.append(distcheck)

        return state

    def manhat_dist(self, x1, y1, x2, y2):
        distance = (x1 - x2) + (y1 - y2)
        return distance / self.max_manhat_dist
    
    def normalize(self, value, min_value, max_value):
        range = max_value - min_value if max_value - min_value != 0 else 1
        return (value - min_value) / range

