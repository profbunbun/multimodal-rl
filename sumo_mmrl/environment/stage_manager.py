

class StageManager:
    def __init__(self, finder, edge_position, sumo):

        self.finder = finder
        self.edge_position = edge_position
        self.sumo = sumo
        self.stage = "pickup"

    def update_stage(self, current_stage, destination_edge, vedge, person, vehicle):

        new_stage = current_stage
        
        new_destination_edge = destination_edge

        if vedge == destination_edge:

            if current_stage == "pickup":
                new_stage = "picked up"
                
                new_destination_edge = self.finder.find_begin_stop(
                    person.get_road(), self.edge_position, self.sumo
                ).partition("_")[0]
                # print(new_stage)

            elif current_stage == "picked up":

                end_stop = self.finder.find_end_stop(
                    person.destination, self.edge_position, self.sumo
                ).partition("_")[0]
                new_destination_edge = end_stop
                vehicle.teleport(new_destination_edge)

                self.sumo.simulationStep()
                new_destination_edge = person.destination
                new_stage = "final"
                # print(new_stage)

            elif current_stage == "final":
                new_stage = "done"
                # print(new_stage)
        return new_stage, new_destination_edge

    def get_initial_stage(self):

        return self.stage
    
    def calculate_reward(self, old_dist, edge_distance, destination_edge, vedge, make_choice_flag, life):
        
        reward = 0
        distcheck = 0

        if old_dist > edge_distance:
            reward = 0.025
            distcheck = 1
        elif old_dist < edge_distance:
            reward = -0.02
            distcheck = 0
        # elif old_dist == edge_distance:
        #     reward = 0.01
        #     distcheck = 0

        if vedge == destination_edge:
            life += 0.1
            reward = 0.8
            make_choice_flag = True

        return reward, make_choice_flag, distcheck, life
