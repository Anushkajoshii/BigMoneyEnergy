# # app/utils.py
# import plotly.graph_objs as go
# import numpy as np

# def fan_chart_figure(final_savings_array, baseline_savings, title="Monte Carlo final savings"):
#     """
#     final_savings_array: 1D numpy array of final savings from sims
#     baseline_savings: current savings (to show a vertical line)
#     """
#     arr = np.array(final_savings_array)
#     p10, p25, p50, p75, p90 = np.percentile(arr, [10,25,50,75,90])
#     hist = go.Histogram(x=arr, nbinsx=50, name="Simulated final savings", opacity=0.9)
#     vline = go.Scatter(x=[baseline_savings, baseline_savings], y=[0, max(np.histogram(arr, bins=50)[0])*1.05],
#                        mode="lines", name="Current savings", line=dict(width=2, dash="dash"))
#     layout = go.Layout(title=title, xaxis=dict(title="Final savings after horizon"), yaxis=dict(title="Count"))
#     fig = go.Figure(data=[hist, vline], layout=layout)
#     return fig


# app/utils.py
import io
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def fan_chart_figure(final_savings_array, baseline_savings, title="Monte Carlo final savings"):
    arr = np.array(final_savings_array)
    hist = go.Histogram(x=arr, nbinsx=50, name="Simulated final savings", opacity=0.9)
    vline = go.Scatter(x=[baseline_savings, baseline_savings], y=[0, max(np.histogram(arr, bins=50)[0])*1.05],
                       mode="lines", name="Current savings", line=dict(width=2, dash="dash"))
    layout = go.Layout(title=title, xaxis=dict(title="Final savings after horizon"), yaxis=dict(title="Count"))
    fig = go.Figure(data=[hist, vline], layout=layout)
    return fig

def create_excel_report(snapshot: dict, simulation_stats: dict, filename="finance_report.xlsx"):
    """
    Returns bytes of an Excel file containing snapshot and simulation summary.
    """
    with io.BytesIO() as output:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame([snapshot]).T.rename(columns={0:"value"}).to_excel(writer, sheet_name="snapshot")
            pd.DataFrame([simulation_stats]).T.rename(columns={0:"value"}).to_excel(writer, sheet_name="simulation")
        return output.getvalue()

def create_pdf_report(snapshot: dict, simulation_stats: dict, product_name: str, horizon_months: int):
    """
    Creates a simple PDF report (bytes) summarizing the input snapshot and simulation stats.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 0.7 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, height - margin, f"GenZ Finance Advisor — Report for {product_name}")

    c.setFont("Helvetica", 10)
    y = height - margin - 30
    c.drawString(margin, y, f"Horizon (months): {horizon_months}")
    y -= 20

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Snapshot")
    y -= 16
    c.setFont("Helvetica", 10)
    for k, v in snapshot.items():
        c.drawString(margin + 10, y, f"{k}: {v}")
        y -= 12
        if y < 80:
            c.showPage()
            y = height - margin

    y -= 6
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Simulation Summary")
    y -= 16
    c.setFont("Helvetica", 10)
    for k, v in simulation_stats.items():
        c.drawString(margin + 10, y, f"{k}: {v}")
        y -= 12
        if y < 80:
            c.showPage()
            y = height - margin

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()
