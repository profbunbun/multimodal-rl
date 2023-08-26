"""import stuff"""
import sys
import os
import xml.etree.ElementTree as ET
import traci

import sumolib

import libsumo

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("No environment variable SUMO_HOME!")


class SUMOConnection:
    """_summary_

    Returns:
        _type_: _description_
    """

    CONNECTION_LABEL = 0

    def __init__(self, sumocfg: str) -> None:
        self.label = str(SUMOConnection.CONNECTION_LABEL)
        SUMOConnection.CONNECTION_LABEL += 1
        self.sumocfg = sumocfg
        self.sumo_=None
        self.sumo_cmd=None

    # do i need the dconnection label?

    def connect_gui(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        self.sumo_cmd = [
            "sumo-gui",
            "-c",
            self.sumocfg,
            "--start",
            "--quit-on-end",
            "--human-readable-time",
        ]
        traci.start(self.sumo_cmd, label=self.label)
        self.sumo_ = traci
        return self.sumo_

    def connect_libsumo_no_gui(self):
        """
        connect_libsumo_no_gui _summary_

        _extended_summary_

        Returns:
            _description_
        """
        self.sumo_cmd = [
            "sumo",
            "-c",
            self.sumocfg,
        ]
        libsumo.start(self.sumo_cmd, label=self.label)
        self.sumo_ = libsumo
        return self.sumo_

    def connect_no_gui(self):
        """
        connect_no_gui _summary_

        _extended_summary_

        Returns:
            _description_
        """
        self.sumo_cmd = [
            "sumo",
            "-c",
            self.sumocfg,
        ]
        traci.start(self.sumo_cmd, label=self.label)
        self.sumo_ = traci
        return self.sumo_

    def close(self):
        """
        close _summary_

        _extended_summary_
        """
        self.sumo_.close()

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
        net = sumolib.net.readNet(path_[0] + "/" + net_file)
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


# con=SUMOConnection("Nets/3x3.sumocfg",False)
# con.connect()
# out_dict, index_dict, edge_list=con.getEdgesInfo()
# print(out_dict, index_dict, edge_list)
# con.close()
