class StepManager:
    def __init__(self, sumo_interface):
        self.sumo_interface = sumo_interface

    def null_step(self, vehicle, make_choice_flag, old_edge):
        """
        Perform null steps in the simulation until a decision-making point is reached.

        Parameters:
        vehicle: The vehicle object to check the current road.
        make_choice_flag: Flag indicating if it's time to make a decision.
        old_edge: The previous edge to compare with the current edge.

        Returns:
        make_choice_flag: Updated flag indicating if a decision can be made.
        old_edge: The edge where the vehicle is currently located.
        """
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
        """
        Perform a simulation step with the given action.

        Parameters:
        vehicle: The vehicle object to apply the action to.
        action: The action to be taken by the vehicle.
        """
        best_choice = vehicle.set_destination(action, destination_edge)
        self.sumo_interface.simulationStep()
        return best_choice
