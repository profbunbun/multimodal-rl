# Graph Components for Dashboard
import plotly.graph_objs as go
from utils.data_loader import smooth_data  # Update the import path as necessary

def create_dual_axis_line_chart(df, x_column, y1_column, y2_column, title):
    fig = go.Figure()

 
    smoothed_rewards = smooth_data(df[y1_column], window_size=100)

    fig.add_trace(go.Scatter(x=df[x_column], y=smoothed_rewards, name=f"Smoothed Average {y1_column}",
                             mode='lines', line=dict(color='red', width=2)))


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

def create_simple_line_chart(df, x_column, y_column, title):
    fig = go.Figure()

    # Add traces for the line chart
    fig.add_trace(go.Scatter(x=df[x_column], y=df[y_column], name=y_column,
                             mode='lines', line=dict(color='green')))

    fig.update_layout(
        title=title,
        xaxis_title=x_column,
        yaxis_title=y_column,
        yaxis=dict(side='left', title=y_column),
        legend=dict(x=0.01, y=0.99, borderwidth=1)
    )

    return fig