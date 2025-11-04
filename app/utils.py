# app/utils.py
import io
import math
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go


# ------------------------------
# Financial Calculations
# ------------------------------

def calculate_emi(principal: float, rate: float, months: int) -> float:
    """
    Calculate monthly EMI for a loan.
    :param principal: Loan amount
    :param rate: Annual interest rate in percent
    :param months: Loan term in months
    :return: Monthly EMI
    """
    if rate == 0:
        return principal / months
    r = rate / (12 * 100)
    emi = principal * r * ((1 + r) ** months) / (((1 + r) ** months) - 1)
    return math.ceil(emi)

# ------------------------------
# Visualization
# ------------------------------

# --------------------------
# FAN CHART (Matplotlib)
# --------------------------
def fan_chart_figure(sim_matrix, months_ahead):
    median = np.median(sim_matrix, axis=0)
    p10 = np.percentile(sim_matrix, 10, axis=0)
    p90 = np.percentile(sim_matrix, 90, axis=0)

    fig, ax = plt.subplots(figsize=(6, 3))
    months = np.arange(months_ahead)

    ax.plot(months, median, label="Median Savings", color="blue")
    ax.fill_between(months, p10, p90, color="blue", alpha=0.2, label="10–90% range")

    ax.set_title("💰 Monte Carlo Savings Projection")
    ax.set_xlabel("Months Ahead")
    ax.set_ylabel("Estimated Savings (₹)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# -----------------------------
# FINAL SAVINGS HISTOGRAM (Plotly)
# -----------------------------
def final_savings_histogram(final_savings_array, baseline_savings):
    arr = np.array(final_savings_array)
    hist = go.Histogram(
        x=arr,
        nbinsx=50,
        name="Simulated Final Savings",
        opacity=0.9
    )
    vline = go.Scatter(
        x=[baseline_savings, baseline_savings],
        y=[0, max(np.histogram(arr, bins=50)[0]) * 1.05],
        mode="lines",
        name="Current Savings",
        line=dict(width=2, dash="dash")
    )
    layout = go.Layout(
        title="🎯 Final Savings Distribution",
        xaxis=dict(title="Final Savings (₹)"),
        yaxis=dict(title="Frequency")
    )
    fig = go.Figure(data=[hist, vline], layout=layout)
    return fig

# ------------------------------
# Report Generation
# ------------------------------

def create_excel_report(snapshot: dict, simulation_stats: dict) -> bytes:
    """
    Generate Excel report from snapshot and simulation stats.
    :return: Bytes of Excel file
    """
    with io.BytesIO() as output:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame([snapshot]).T.rename(columns={0: "value"}).to_excel(writer, sheet_name="snapshot")
            pd.DataFrame([simulation_stats]).T.rename(columns={0: "value"}).to_excel(writer, sheet_name="simulation")
        return output.getvalue()



def create_pdf_report(snapshot: dict, simulation_stats: dict, product_name: str, horizon_months: int, recommendation: str = "No recommendation available.") -> bytes:
    """
    Generate a well-formatted PDF report summarizing user finances, simulation results, and recommendation.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    margin = 0.8 * inch
    y = height - margin

    # === HEADER ===
    c.setFont("Helvetica-Bold", 18)
    c.setFillColor(colors.darkblue)
    c.drawString(margin, y, "💸 GenZ Finance Advisor Report")
    c.setFillColor(colors.black)
    c.setFont("Helvetica", 10)
    c.drawString(width - 200, y, datetime.now().strftime("Generated on %b %d, %Y"))
    y -= 25

    # === PRODUCT SUMMARY ===
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, f"Purchase Summary — {product_name}")
    y -= 15
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Horizon: {horizon_months} months")
    y -= 15
    # c.drawString(margin, y, f"Price: ₹{snapshot.get('car_price', 'N/A'):,}")
    price = snapshot.get("car_price")
    price_str = f"₹{price:,}" if isinstance(price, (int, float)) else "N/A"
    c.drawString(margin, y, f"Price: {price_str}")

    y -= 25

    # === FINANCIAL SNAPSHOT ===
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "📊 Financial Snapshot")
    y -= 18
    c.setFont("Helvetica", 10)
    for k, v in snapshot.items():
        val_str = f"₹{v:,.0f}" if isinstance(v, (int, float)) else str(v)
        c.drawString(margin + 10, y, f"{k.replace('_', ' ').title()}: {val_str}")
        y -= 12
        if y < 80:
            c.showPage()
            y = height - margin

    # === SIMULATION RESULTS ===
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "🎲 Simulation Summary")
    y -= 18
    c.setFont("Helvetica", 10)
    for k, v in simulation_stats.items():
        val = f"{v:.3f}" if isinstance(v, float) else str(v)
        c.drawString(margin + 10, y, f"{k.replace('_', ' ').title()}: {val}")
        y -= 12
        if y < 80:
            c.showPage()
            y = height - margin

    # === RECOMMENDATION SECTION ===
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.darkgreen if "consider" in recommendation.lower() else colors.darkred)
    c.drawString(margin, y, "🧠 Recommendation")
    c.setFillColor(colors.black)
    y -= 15
    c.setFont("Helvetica", 10)
    wrapped_text = simpleSplit(recommendation, "Helvetica", 10, width - 2 * margin)
    for line in wrapped_text:
        c.drawString(margin, y, line)
        y -= 12
        if y < 80:
            c.showPage()
            y = height - margin

    # === FOOTER ===
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.grey)
    c.drawCentredString(width / 2, 40, "Generated by GenZ Finance Advisor • Not financial advice")

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

