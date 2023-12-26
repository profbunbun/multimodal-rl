# app.py
from dash import Dash, dcc, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import os

EXPERIMENT_PATH = 'Experiments/3x3/logger/'



def list_dirs(directory):
    try:
        return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    except FileNotFoundError:
        return []

folder_list = list_dirs(EXPERIMENT_PATH)



app = Dash(__name__, external_stylesheets=[dbc.themes.VAPOR])
#components
title = dcc.Markdown(children='')
graph1 = dcc.Graph(figure={})
dropdown = dcc.Dropdown(options=[{'label': name, 'value': name} for name in folder_list], value=folder_list[0] if folder_list else None, id='dropdown')
data_paths_store = dcc.Store(data=None, id='data_paths_store')


#layout
app.layout = dbc.Container([
    dbc.Row([dbc.Col(title, width=12)],justify='center'),
    dbc.Row([dbc.Col(dropdown, width=12)]),
    dbc.Row([dbc.Col(graph1, width=6)]),
    data_paths_store
    ])
                                   

#callbacks
@app.callback(Output(data_paths_store, 'data'), [Input('dropdown', 'value')])


def update_data_paths(value):
    if not value:
        return {}  # Return an empty dictionary if value is None

    model_info = None

    # Construct the path for each CSV file
    base_path = os.path.join(EXPERIMENT_PATH, value)
    # Open the file for reading

    paths = {
        'training': os.path.join(base_path, 'training_log.csv'),
        'config': os.path.join(base_path, 'config_log.csv'),
        'episodes': os.path.join(base_path, 'episode_log.csv'),
        'steps': os.path.join(base_path, 'step_log.csv'),
        'model_info': os.path.join(base_path, 'model_info.txt')
    }

    with open(paths['model_info'], 'r') as f:
        model_info = f.read()
    # Read the data from CSV files and convert them to JSON serializable format
    data = {
        'model_info': model_info,
        'training_df': pd.read_csv(paths['training']).to_dict(orient='records'),
        'config_df': pd.read_csv(paths['config']).to_dict(orient='records'),
        'episodes_df': pd.read_csv(paths['episodes']).to_dict(orient='records'),
        'step_df': pd.read_csv(paths['steps']).to_dict(orient='records')
    }

    return data

# @app.callback(Output(graph1, 'figure'), [Input(data_paths_store, 'data')])


if __name__ == '__main__':
    app.run_server(debug=True)
