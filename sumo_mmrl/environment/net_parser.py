"""import stuff"""
import xml.etree.ElementTree as ET
import sumolib



class NetParser():
    '''
    NetParser _summary_

    _extended_summary_
    '''    
    def __init__(self,sumocfg)->None:
        self.sumocfg = sumocfg

    def parse_net_files(self):
        """
        parse_net_files _summary_

        _extended_summary_

        Returns:
            _description_
        """
        tree = ET.parse(self.sumocfg)
        root = tree.getroot()
        for infile in root.findall("input"):
            for network in infile.findall("net-file"):
                network_file = str(network.get("value"))
            return network_file

    def get_edges_info(self):
        """
        getEdgesInfo _summary_

        _extended_summary_

        Returns:
            _description_
        """
        net_file = self.parse_net_files()
        path_ = self.sumocfg.rsplit("/")
        path_.pop()
        path_b = "/".join(path_)
        net = sumolib.net.readNet(path_b + "/" + net_file)
        out_dict = {}
        length_dict = {}
        index_dict = {}
        edge_list = []
        edge_position_dict = {}
        counter = 0
        all_edges = net.getEdges()
        for current_edge in all_edges:
            current_edge_id = current_edge.getID()

            if current_edge_id in edge_position_dict:
                print(current_edge_id + " already exists!")
            else:
                edge_start, edge_end = current_edge.getShape()
                x = (edge_start[0] + edge_end[0]) / 2

                y = (edge_start[1] + edge_end[1]) / 2

                edge_position_dict[current_edge_id] = x, y

            if current_edge.allows("passenger"):
                edge_list.append(current_edge)

            if current_edge_id in index_dict:
                print(current_edge_id + " already exists!")
            else:
                index_dict[current_edge_id] = counter
                counter += 1
            if current_edge_id in out_dict:
                print(current_edge_id + " already exists!")
            else:
                out_dict[current_edge_id] = {}
            if current_edge_id in length_dict:
                print(current_edge_id + " already exists!")
            else:
                length_dict[current_edge_id] = current_edge.getLength()
            # edge_now is sumolib.net.edge.Edge
            out_edges = current_edge.getOutgoing()
            for current_out_edge in out_edges:
                if not current_out_edge.allows("passenger"):
                    # print("Found some roads prohibited")
                    continue
                conns = current_edge.getConnections(current_out_edge)
                for conn in conns:
                    dir_now = conn.getDirection()
                    out_dict[current_edge_id][dir_now] = current_out_edge.getID()

        return [out_dict, index_dict, edge_list, edge_position_dict]
