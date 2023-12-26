# app.py

from dash import Dash, dcc, Output, Input, State, no_update
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
epsilon = dcc.Graph(figure={}, id='epsilon')
step_plot = dcc.Graph(figure={}, id='step_plot')
dropdown = dcc.Dropdown(options=[{'label': name, 'value': name} for name in folder_list], value=folder_list[0] if folder_list else None, id='dropdown')
data_paths_store = dcc.Store(data=None, id='data_paths_store')


#layout
app.layout = dbc.Container([
    dbc.Row([dbc.Col(title, width=12)],justify='center'),
    dbc.Row([dbc.Col(dropdown, width=12)]),
    dbc.Row([dbc.Col(epsilon, width=6), dbc.Col(step_plot, width=6)]),
    data_paths_store,
    dcc.Interval(id='interval', interval=1000, n_intervals=0)
    ])
                                   

#callbacks
@app.callback(Output(data_paths_store, 'data'), [Input('dropdown', 'value')])
def update_data_paths(value):

    if not value:

        return {} 
    model_info = None

    # Construct the path for each CSV file
    base_path = os.path.join(EXPERIMENT_PATH, value)
    # Open the file for reading

    paths = {
        # 'training': os.path.join(base_path, 'training_log.csv'),
        'config': os.path.join(base_path, 'config_log.csv'),
        'episodes': os.path.join(base_path, 'episode_log.csv'),
        # 'steps': os.path.join(base_path, 'step_log.csv'),
        'model_info': os.path.join(base_path, 'model_info.txt')
    }

    with open(paths['model_info'], 'r') as f:
        model_info = f.read()
    # Read the data from CSV files and convert them to JSON serializable format
    data = {
        'selected_folder': value,
    }
    

    return data


@app.callback([Output('epsilon', 'figure'), Output('step_plot', 'figure')],
              [Input('interval', 'n_intervals')],
              [State(data_paths_store, 'data')])
def update_graphs(n, data):
    if not data or 'selected_folder' not in data:

        return no_update
    
    selected_folder = data.get('selected_folder', '')
    episodes_path = os.path.join(EXPERIMENT_PATH, selected_folder, 'episode_log.csv')



    episodes_df = pd.read_csv(episodes_path)

    episodes_df['smoothed_reward'] = episodes_df['episode_reward'].rolling(window=100, min_periods=1).mean()
    episodes_df['smoothed_steps'] = episodes_df['agent_steps'].rolling(window=100, min_periods=1).mean()
    

    epsilon_fig = px.line(episodes_df, x='episode', y='epsilon', title='Epsilon decay and reward over time')
    epsilon_fig.add_scatter(x=episodes_df['episode'], y=episodes_df['episode_reward'], name='Episode Reward', mode='lines', yaxis='y2')
    epsilon_fig.add_scatter(x=episodes_df['episode'], y=episodes_df['smoothed_reward'], name='Smoothed Reward', mode='lines', yaxis='y2')

    steps_fig = px.line(episodes_df, x='episode', y='smoothed_steps', title='Steps taken over time')
    steps_fig.add_scatter(x=episodes_df['episode'], y=episodes_df['agent_steps'], name='Steps', mode='lines')


    epsilon_fig.update_layout(
        yaxis2=dict(
            title="Reward",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
            overlaying="y",
            side="right"
        )
    )

    steps_fig.update_layout(
        yaxis2=dict(
            title="Steps",
            titlefont=dict(color="blue"),
            tickfont=dict(color="blue"),
            overlaying="y",
            side="right"
        )
    )
    
    return epsilon_fig, steps_fig


if __name__ == '__main__':
    app.run_server(debug=True)
