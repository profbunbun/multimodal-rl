class RewardCalculator:
    def __init__(self, edge_position):
        self.edge_position = edge_position

    def calculate_reward(self, old_dist, edge_distance, stage, destination_edge, vedge, make_choice_flag, life):
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
            life += 10
            if stage == "pickup":
                reward = 0.3

                make_choice_flag = True

            elif stage == "dropoff":
                reward = 0.3

                make_choice_flag = True

            elif stage == "onbus":
                reward = 0.3

                make_choice_flag = True

            elif stage == "final":
                reward = 0.35

                make_choice_flag = True

        return reward, make_choice_flag, distcheck, life
