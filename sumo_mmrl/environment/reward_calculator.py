class RewardCalculator:
    """
    RewardCalculator class for calculating rewards based on various factors.

    :param dict edge_position: Dictionary of edge positions.
    """

    def __init__(self, edge_position):
        """
        Initialize the RewardCalculator with the given edge positions.
        """
        self.edge_position = edge_position

    def calculate_reward(self, old_dist, edge_distance, stage, destination_edge, vedge, make_choice_flag, life):
        """
        Calculate the reward based on distance and other factors.

        :param float old_dist: Previous distance to the destination.
        :param float edge_distance: Current distance to the destination.
        :param str stage: Current stage of the person.
        :param str destination_edge: Destination edge ID.
        :param str vedge: Current vehicle edge ID.
        :param bool make_choice_flag: Flag indicating if a choice is made.
        :param int life: Life points of the person.
        :return: Tuple of reward, updated make_choice_flag, distcheck, and life.
        :rtype: tuple
        """
        reward = 0
        distcheck = 0

        if old_dist > edge_distance:
            reward = 0.025
            distcheck = 1
        # elif old_dist < edge_distance:
        #     reward = -0.02
        #     distcheck = 0
        # elif old_dist == edge_distance:
        #     reward = 0.01
        #     distcheck = 0

        if vedge == destination_edge:
            life += 0.1
            if stage == "pickup":
                reward = 0.8

                make_choice_flag = True

            elif stage == "dropoff":
                reward = 0.8

                make_choice_flag = True

            elif stage == "onbus":
                reward = 0.8

                make_choice_flag = True

            elif stage == "final":
                reward = 0.8

                make_choice_flag = True

        return reward, make_choice_flag, distcheck, life
