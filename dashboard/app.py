import os
import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from components.graph_components import create_dual_axis_line_chart  # Update import path as necessary
from utils.data_loader import load_data, get_directories  # Update import path as necessary
from config.dashboard_config import EXPERIMENT_PATH # Update import path as necessary
from components.control_components import create_file_tree_component

app = dash.Dash(__name__)



app.layout = html.Div([
    # File tree on the left
    html.Div(create_file_tree_component(EXPERIMENT_PATH), className='three columns'),

    # Main content area on the right
    html.Div([
        html.H1('Reinforcement Learning Agent Dashboard'),
        dcc.Graph(id='main-content'),
        dcc.Interval(id='interval-component', interval=1*1000, n_intervals=0)
    ], className='nine columns')
], className='row')

# Callbacks and other functionalities remain the same



@app.callback(
    Output('log-folder-dropdown', 'options'),
    [Input('experiment-dropdown', 'value')]
)
def set_log_folder_options(selected_experiment):
    if selected_experiment is not None:
        log_folders = get_directories(os.path.join(EXPERIMENT_PATH, selected_experiment, '/logger/'))
        return [{'label': name, 'value': name} for name in log_folders]
    return []

@app.callback(
    Output('rewards-epsilon-per-episode', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('log-folder-dropdown', 'value')]
)
def update_graph(n, selected_log_folder):
    if selected_log_folder:
        global SELECTED_PATH
        SELECTED_PATH = os.path.join(EXPERIMENT_PATH, 'logger/', selected_log_folder, 'episode_log.csv')
        df_episode = load_data(SELECTED_PATH)
        rewards_epsilon_figure = create_dual_axis_line_chart(df_episode, 'episode', 'episode_reward', 'epsilon', 'Average Rewards and Epsilon Over Episodes')
        return rewards_epsilon_figure
    return go.Figure()  # Return an empty figure if no log folder is selected

if __name__ == '__main__':
    app.run_server(debug=True)
