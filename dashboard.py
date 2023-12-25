# Import necessary libraries
import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px



FOLDERPATH = 'Experiments/3x3/logger/eps20000_lr0.001_d0.999/'
STEPFILE = 'step_log.csv'

df = pd.read_csv(FOLDERPATH + STEPFILE)

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(children='Dashboard for CSV Logs'),

    dcc.Graph(
        id='example-graph',
        figure=px.line(df, x='episode', y='reward', title='Rewards over Episodes')
    )
])
if __name__ == '__main__':
    app.run_server(debug=True)