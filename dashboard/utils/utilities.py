# Data Loader for Dashboard
import os
import pandas as pd
import xml.etree.ElementTree as ET

# Load and parse the SUMO network file



def smooth_data(data, window_size=100):
    """Smooth data using a simple moving average."""
    if data.empty:  # Check if the Series is empty
        return []

    smoothed = pd.Series(data).rolling(window=window_size, min_periods=1).mean()
    return smoothed

def parse_shape(shape_str):
    return [(float(coord.split(',')[0]), float(coord.split(',')[1])) for coord in shape_str.split(' ')]


def parse_map(path):
    tree = ET.parse(path)
    root = tree.getroot()
    # Initialize a list to hold edge data
    edges_data = []

    # Iterate through each edge in the file
    for edge in root.findall('edge'):
        edge_id = edge.get('id')
        function = edge.get('function')

        # Each edge can have multiple lanes. Store this information too.
        lanes = []
        for lane in edge.findall('lane'):
            lane_id = lane.get('id')
            speed = lane.get('speed')
            length = lane.get('length')
            shape = lane.get('shape')
            

        shape = parse_shape(lane.get('shape'))  # Update to parse the shape string
        lanes.append({'lane_id': lane_id, 'speed': speed, 'length': length, 'shape': shape})

        # Append edge info to the edges_data list
        edges_data.append({
            'id': edge_id,
            'function': function,
            'lanes': lanes
        })
    return edges_data
