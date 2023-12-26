# Import necessary libraries
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go

FOLDERPATH = 'Experiments/3x3/logger/eps10000_lr0.001_d0.998/'
STEPFILE = 'step_log.csv'
EPISODEFILE = 'episode_log.csv'
TRAINIGFILE = 'training_log.csv'

df_episode = pd.read_csv(FOLDERPATH + EPISODEFILE)
def smooth_data(data, window_size=100):
    """Smooth data using a simple moving average."""
    if data.empty:  # Check if the Series is empty
        return []

    smoothed = pd.Series(data).rolling(window=window_size, min_periods=1).mean()
    return smoothed

app = dash.Dash(__name__)

def create_dual_axis_line_chart(df, x_column, y1_column, y2_column, title):
    fig = go.Figure()

    # Calculate the smoothed average rewards
    smoothed_rewards = smooth_data(df[y1_column], window_size=100)

    # Add traces for the original rewards
    # fig.add_trace(go.Scatter(x=df[x_column], y=df[y1_column], name=y1_column,
    #                          mode='lines', line=dict(color='orange', width=1), opacity=0.6))

    # Add traces for the smoothed rewards
    fig.add_trace(go.Scatter(x=df[x_column], y=smoothed_rewards, name=f"Smoothed Average {y1_column}",
                             mode='lines', line=dict(color='red', width=2)))

    # Add traces for epsilon
    fig.add_trace(go.Scatter(x=df[x_column], y=df[y2_column], name=y2_column,
                             mode='lines', line=dict(color='blue'), yaxis='y2'))

    fig.update_layout(
        title_text=title,
        xaxis=dict(title=x_column),
        yaxis=dict(title=y1_column, side='left'),
        yaxis2=dict(
            title=y2_column,
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue'),
            overlaying='y',
            side='right'
        )
    )

    return fig

@app.callback(Output('rewards-epsilon-per-episode', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph(n):
    df_episode = pd.read_csv(FOLDERPATH + EPISODEFILE)
    figure = create_dual_axis_line_chart(df_episode, 'episode', 'episode_reward', 'epsilon', 'Average Rewards and Epsilon Over Episodes')
    return figure

app.layout = html.Div(children=[
    html.H1(children='Reinforcement Learning Agent Dashboard'),
    dcc.Graph(id='rewards-epsilon-per-episode'),
    dcc.Interval(
            id='interval-component',
            interval=1*1000,  # Update every 1 second
            n_intervals=0
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)