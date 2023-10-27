
import xml.etree.ElementTree as ET
import sumolib


class NetParser:

    def __init__(self, sumocfg) -> None:
        self.sumocfg = sumocfg

    def parse_net_files(self):
        '''get net file from the sumo config file '''

        tree = ET.parse(self.sumocfg)
        root = tree.getroot()
        for infile in root.findall("input"):
            for network in infile.findall("net-file"):
                network_file = str(network.get("value"))
            return network_file

    def _clean_path(self):
        '''clean file path '''

        net_file = self.parse_net_files()
        path_ = self.sumocfg.rsplit("/")
        path_.pop()
        path_b = "/".join(path_)
        return sumolib.net.readNet(path_b + "/" + net_file)

    def get_edges_info(self):
        ''' gets list of edges'''

        net = self._clean_path()
        edge_list = []
        all_edges = net.getEdges()
        for current_edge in all_edges:
            if current_edge.allows("passenger"):
                edge_list.append(current_edge)
        return edge_list

    def get_edge_pos_dic(self):
        ''' gets xy coords of the center of the edge'''

        net = self._clean_path()
        edge_position_dict = {}
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
        return edge_position_dict

    def get_out_dic(self):
        ''' gets dic of connecting edges for each edge'''

        net = self._clean_path()
        out_dict = {}
        all_edges = net.getEdges()
        for current_edge in all_edges:
            current_edge_id = current_edge.getID()
            if current_edge_id in out_dict:
                print(current_edge_id + " already exists!")
            else:
                out_dict[current_edge_id] = {}
            out_edges = current_edge.getOutgoing()
            for current_out_edge in out_edges:
                if not current_out_edge.allows("passenger"):
                    # print("Found some roads prohibited")
                    continue
                conns = current_edge.getConnections(current_out_edge)
                for conn in conns:
                    dir_now = conn.getDirection()
                    out_dict[
                        current_edge_id][dir_now] = current_out_edge.getID()
        return out_dict

    def get_edge_index(self):
        '''indexed dic of edge ids'''

        net = self._clean_path()
        index_dict = {}
        counter = 0
        all_edges = net.getEdges()
        for current_edge in all_edges:
            current_edge_id = current_edge.getID()

            if current_edge_id in index_dict:
                print(current_edge_id + " already exists!")
            else:
                index_dict[current_edge_id] = counter
                counter += 1
        return index_dict

    def get_length_dic(self):

        net = self._clean_path()

        length_dict = {}
        all_edges = net.getEdges()
        for current_edge in all_edges:
            current_edge_id = current_edge.getID()
            if current_edge_id in length_dict:
                print(current_edge_id + " already exists!")
            else:
                length_dict[current_edge_id] = current_edge.getLength()
        return length_dict

    def get_route_edges(self):
        edge_ids = []
        for route in sumolib.xml.parse_fast("Experiments/3x3/Nets/3x3_2.rou.xml", 'route', ['id','edges']):
            if 'bus' in route.id:
                edge_ids = route.edges.split()
        # print (edge_ids)
        return edge_ids

    def get_max_manhattan(self):
        a=self.get_edge_pos_dic()
        a=list(a.values())
        n=len(a)
        
        V = [0 for i in range(n)]
        V1 = [0 for i in range(n)]
 
        for i in range(n):
            V[i] = a[i][0] + a[i][1]
            V1[i] = [i][0] - a[i][1]
 
        # Sorting both the vectors
        V.sort()
        V1.sort()
    
        maximum = max(V[-1] - V[0],
                    V1[-1] - V1[0])
 
        # print(maximum)

        return maximum
