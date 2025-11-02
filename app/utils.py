# app/utils.py
import plotly.graph_objs as go
import numpy as np

def fan_chart_figure(final_savings_array, baseline_savings, title="Monte Carlo final savings"):
    """
    final_savings_array: 1D numpy array of final savings from sims
    baseline_savings: current savings (to show a vertical line)
    """
    arr = np.array(final_savings_array)
    p10, p25, p50, p75, p90 = np.percentile(arr, [10,25,50,75,90])
    hist = go.Histogram(x=arr, nbinsx=50, name="Simulated final savings", opacity=0.9)
    vline = go.Scatter(x=[baseline_savings, baseline_savings], y=[0, max(np.histogram(arr, bins=50)[0])*1.05],
                       mode="lines", name="Current savings", line=dict(width=2, dash="dash"))
    layout = go.Layout(title=title, xaxis=dict(title="Final savings after horizon"), yaxis=dict(title="Count"))
    fig = go.Figure(data=[hist, vline], layout=layout)
    return fig
