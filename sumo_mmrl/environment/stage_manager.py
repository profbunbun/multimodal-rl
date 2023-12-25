class StageManager:
    def __init__(self, finder, edge_position, sumo, bussroute):
        self.finder = finder
        self.edge_position = edge_position
        self.sumo = sumo
        self.bussroute = bussroute
        self.stage = "pickup"

    def update_stage(self, current_stage, destination_edge, vedge, person):
        new_stage = current_stage
        
        new_destination_edge = destination_edge

        if vedge == destination_edge:
            # If the vehicle is at the destination edge, update the stage
            if current_stage == "pickup":
                new_stage = "dropoff"
                
                new_destination_edge = self.finder.find_begin_stop(
                    person.get_road(), self.edge_position, self.sumo
                ).partition("_")[0]
                print(new_stage)

            elif current_stage == "dropoff":
                new_stage = "onbus"
                next_route_edge = self.bussroute[1].partition("_")[0]
                new_destination_edge = next_route_edge
                print(new_stage)

            elif current_stage == "onbus":
                new_stage = "final"
                end_stop = self.finder.find_end_stop(
                    person.destination, self.edge_position, self.sumo
                ).partition("_")[0]
                new_destination_edge = end_stop
                print(new_stage)

            elif current_stage == "final":
                new_stage = "done"
                print(new_stage)
        return new_stage, new_destination_edge

    def get_initial_stage(self):
        return self.stage
