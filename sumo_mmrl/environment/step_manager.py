class StepManager:
    
    def __init__(self, sumo_interface):

        self.sumo_interface = sumo_interface

    def null_step(self, vehicle, make_choice_flag, old_edge):

        vedge = vehicle.get_road()

        while not make_choice_flag:
            self.sumo_interface.simulationStep()
            vedge = vehicle.get_road()

            if (":" in vedge) or (old_edge == vedge):
                make_choice_flag = False
            else:
                make_choice_flag = True
            old_edge = vedge

        return make_choice_flag, old_edge

    def perform_step(self, vehicle, action, destination_edge):
   
        target = vehicle.set_destination(action, destination_edge)
        vehicle.teleport(target)
        self.sumo_interface.simulationStep()
        vedge = vehicle.get_road()
        return  vedge
