
class OutMask:

    def get_outmask(self, vedge, pedge, choices, edge_position):

        vedge_loc = edge_position[vedge]
        pedge_loc = edge_position[pedge.partition("_")[0]]
        edge_distance = self.manhat_dist(
            vedge_loc[0], vedge_loc[1], pedge_loc[0], pedge_loc[1]
        )
        outmask = [-1, -1, -1, -1]

        for key, value in choices.items():
            if key == "r":
                sloc = edge_position[value]
                s_dist = self.manhat_dist(sloc[0], sloc[1], pedge_loc[0],
                                          pedge_loc[1])
                if s_dist < edge_distance:
                    outmask[0] = 1

            elif key == "s":
                tloc = edge_position[value]
                t_dist = self.manhat_dist(tloc[0], tloc[1], pedge_loc[0],
                                          pedge_loc[1])
                if t_dist < edge_distance:
                    outmask[1] = 1

            elif key == "l":
                rloc = edge_position[value]
                r_dist = self.manhat_dist(rloc[0], rloc[1], pedge_loc[0],
                                          pedge_loc[1])
                if r_dist < edge_distance:
                    outmask[2] = 1

            elif key == "t":
                lloc = edge_position[value]
                l_dist = self.manhat_dist(lloc[0], lloc[1], pedge_loc[0],
                                          pedge_loc[1])
                if l_dist < edge_distance:
                    outmask[3] = 1

        return vedge_loc, pedge_loc, outmask, edge_distance

    def get_outmask_valid(self, choices):

        outmask = [0, 0, 0, 0]

        for choice in choices.items():
            if choice[0] == "r":
                outmask[0] = 1

            elif choice[0] == "s":
                outmask[1] = 1

            elif choice[0] == "l":
                outmask[2] = 1

            elif choice[0] == "t":
                outmask[3] = 1

        return outmask

    def manhat_dist(self, x1, y1, x2, y2):
      
        return abs(x1 - x2) + abs(y1 - y2)
